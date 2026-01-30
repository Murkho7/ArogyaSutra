import os
import sys
import time
import io
import tempfile
import numpy as np
import torch
import multiprocessing as mp
import matplotlib.pyplot as plt
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

# # Add path
# sys.path.append(os.path.join(os.getcwd(), "LLMDet"))

# Import HuggingFace related libraries
from transformers import GroundingDinoProcessor
from modeling_grounding_dino import GroundingDinoForObjectDetection

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Global variables - model worker pool
worker_pool = []
worker_index = 0

# Use asynccontextmanager instead of on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_pool, worker_index
    
    # Detect number of available GPUs
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("No GPU devices available, will use CPU")
        gpu_count = 1  # Use CPU
    else:
        print(f"Detected {gpu_count} GPU devices")
    
    # Load configuration
    # This can be loaded from environment variables or config files
    default_config = {
        "model_id": "fushh7/llmdet_swin_large_hf",  # HF model ID
        "box_threshold": 0.4,
        "text_threshold": 0.3,
    }
    
    # Create a model worker process for each GPU
    for gpu_id in range(gpu_count):
        device = f"cuda:{gpu_id}" if gpu_count > 1 else ("cuda:0" if gpu_count == 1 and torch.cuda.is_available() else "cpu")
        default_config["device"] = device
        worker = GroundingDINOWorker(**default_config)
        worker.start()
        worker_pool.append(worker)
    
    print(f"Initialized model worker processes on {len(worker_pool)} devices")
    
    # Code to run during application lifetime
    yield
    
    # Cleanup code when application shuts down
    print("Shutting down all model worker processes")
    for worker in worker_pool:
        worker.stop()
    worker_pool = []

# Create FastAPI app with lifespan
app = FastAPI(
    title="LLMDet API",
    description="API for LLMDet object detection",
    version="1.0.0",
    lifespan=lifespan
)

class GroundingDINOWorker:
    def __init__(self, model_id, box_threshold=0.4, text_threshold=0.3, device="cpu"):
        """Initialize model worker process"""
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        
        self.request_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.process = None
        
    def start(self):
        """Start the model worker process"""
        self.process = mp.Process(
            target=self._worker_process, 
            args=(
                self.model_id, self.box_threshold, self.text_threshold, 
                self.device, self.request_queue, self.result_queue
            )
        )
        self.process.daemon = True
        self.process.start()
        
    def _worker_process(self, model_id, box_threshold, text_threshold, 
                        device, request_queue, result_queue):
        """Worker process function: load model and handle requests"""
        try:
            # Set device
            if device.startswith("cuda:"):
                gpu_id = int(device.split(":")[-1])
                torch.cuda.set_device(gpu_id)
            
            # Initialize LLMDet model
            print(f"Loading LLMDet model to {device}...")
            processor = GroundingDinoProcessor.from_pretrained(model_id)
            model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)
            
            print(f"Model loaded to {device}")
            
            # Add a counter to clean memory every N requests
            request_count = 0
            clean_interval = 10  # Clean once every 10 requests

            # Request handling loop
            while True:
                try:
                    # Get request
                    request_id, request_data = request_queue.get()
                    
                    # Process detection request
                    image_pil, text_prompt = request_data
                    
                    # Use HuggingFace processor and model to perform inference
                    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        target_sizes=[image_pil.size[::-1]]  # [height, width]
                    )[0]  # take the first result (single image)
                    
                    # Check if any objects detected
                    if len(results["boxes"]) == 0:
                        print(f"No '{text_prompt}' detected.")
                        result = {"message": f"No '{text_prompt}' detected"}
                        result_queue.put((request_id, result))
                        
                        # --- ADDED: Clean up tensors even when no objects are detected ---
                        del inputs, outputs, results
                        # -------------------------------------------------------------
                            
                        continue  # handle next request
                    
                    # Prepare returned object info
                    detected_objects = []
                    
                    # Gather detected object info
                    boxes = results["boxes"].cpu().numpy()
                    scores = results["scores"].cpu().numpy()
                    labels = results["labels"]
                    
                    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                        detected_objects.append({
                            "id": i + 1,  # 1-based index
                            "label": label,
                            "bbox": box.tolist(),  # [x_min, y_min, x_max, y_max]
                            "confidence": float(score)
                        })
                    
                    # Generate visualization result
                    result_image = self._generate_visualization(
                        np.array(image_pil), 
                        boxes, 
                        [str(i+1) for i in range(len(boxes))]  # use indices as labels
                    )
                    
                    # Save the result as a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        plt.imsave(tmp.name, result_image)
                        tmp_filename = tmp.name
                            
                    # Return result
                    result = {
                        "visualization_path": tmp_filename,
                        "detected_objects": detected_objects
                    }
                        
                    result_queue.put((request_id, result))

                except Exception as e:
                    print(f"{device} worker process error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    result_queue.put((request_id, {"error": str(e)}))

                finally:
                    # --- ADDED: Explicitly delete tensors and clear cache after every request ---
                    if 'inputs' in locals():
                        del inputs
                    if 'outputs' in locals():
                        del outputs
                    if 'results' in locals():
                        del results

                    import gc; gc.collect()

                    if request_count % clean_interval == 0:
                        torch.cuda.empty_cache()

        except Exception as e:
            print(f"{device} initialization failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _generate_visualization(self, image, boxes, labels):
        """Generate visualization result
        
        Parameters:
        - image: original image (numpy array)
        - boxes: bounding boxes
        - labels: labels (indices)
        """
        # Set figure size based on original image dimensions
        h, w = image.shape[:2]
        dpi = 100  # DPI setting
        figsize = (w/dpi, h/dpi)  # compute figsize from image dims and DPI
        
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(image)
        
        # Create a random color for each label
        label_colors = {}
        for label in labels:
            # Use random colors instead of fixed colors based on index
            rgb = np.random.random(3)  # random RGB values
            label_colors[label] = np.concatenate([rgb, np.array([1])], axis=0)
        
        # Display bounding boxes
        for box, label in zip(boxes, labels):
            color = label_colors[label]
            self._show_box(box, plt.gca(), label, color)
        
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0) # Remove margins
        
        # Convert figure to image
        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()
        import gc; gc.collect()
        
        return img_array
        
    def _show_box(self, box, ax, label, color):
        """Show box and label"""
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]

        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))
        
        # Show label with larger font and background matching box color
        text = ax.text(x0, y0-5, label, fontsize=12, color='white', 
                    bbox=dict(facecolor=color, alpha=0.8, pad=2))

    def process_request(self, image_pil, text_prompt):
        """Submit a request to the worker process and wait for the result"""
        request_id = f"{time.time()}_{id(image_pil)}"
        self.request_queue.put((request_id, (image_pil, text_prompt)))
        
        # Wait for result
        while True:
            try:
                result_id, result = self.result_queue.get(timeout=60)  # 60s timeout
                if result_id == request_id:
                    return result
            except Exception as e:
                raise RuntimeError(f"Timeout or error while processing request: {str(e)}")
                
    def stop(self):
        """Stop the worker process"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()

def get_next_worker():
    """Simple round-robin scheduling to get the next worker process"""
    global worker_index, worker_pool
    
    if not worker_pool:
        raise RuntimeError("No available model worker processes")
    
    worker = worker_pool[worker_index]
    worker_index = (worker_index + 1) % len(worker_pool)
    return worker

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    text_prompt: str = Form(...),
    box_threshold: Optional[float] = Form(0.4),
    text_threshold: Optional[float] = Form(0.3)
):
    """
    Upload an image and text prompt for object detection
    
    Parameters:
    - file: uploaded image file
    - text_prompt: text description specifying objects to detect
    - box_threshold: bounding box confidence threshold
    - text_threshold: text matching threshold
    
    Returns:
    - visualization image path and detailed detected objects information
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    try:
        # Read uploaded image
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents))
        # Convert to RGB to ensure correct format
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        
        # Get a worker and process the request
        worker = get_next_worker()
        result = worker.process_request(image_pil, text_prompt)
        
        # Check result type
        if "message" in result:
            # If no objects detected, return JSON message
            return JSONResponse(content=result, status_code=200)
        
        if "error" in result:
            raise Exception(result["error"])
        
        # Return visualization path and detected objects info
        visualization_path = result["visualization_path"]
        detected_objects = result.get("detected_objects", [])
        
        return JSONResponse(content={
            "visualization_path": visualization_path,
            "detected_objects": detected_objects
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_count = torch.cuda.device_count()
    active_workers = len([w for w in worker_pool if w.process and w.process.is_alive()])
    
    return {
        "status": "healthy" if active_workers == len(worker_pool) else "degraded",
        "gpu_count": gpu_count,
        "active_workers": active_workers,
        "total_workers": len(worker_pool)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)