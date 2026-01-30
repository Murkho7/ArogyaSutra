import io
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
import matplotlib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, FileResponse, JSONResponse
import tempfile
import base64
import os
import queue
from pydantic import BaseModel
from typing import Dict, List, Optional
import time
from contextlib import asynccontextmanager

from depth_anything_v2.dpt import DepthAnythingV2

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Global variable - Model worker process pool
worker_pool = []
worker_index = 0

# Use asynccontextmanager instead of on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_pool, worker_index
    
    # Detect available GPU count
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No GPU devices available")
    
    print(f"Detected {gpu_count} GPU devices")
    
    # Create a model worker process for each GPU
    model_type = 'vitl'  # Can change default model type as needed
    for gpu_id in range(gpu_count):
        worker = ModelWorker(gpu_id=gpu_id, model_type=model_type)
        worker.start()
        worker_pool.append(worker)
    
    print(f"Initialized model worker processes on {len(worker_pool)} GPUs")
    
    # Application runtime code
    yield
    
    # Cleanup code during application shutdown
    print("Shutting down all model worker processes")
    for worker in worker_pool:
        worker.stop()
    worker_pool = []

# Create FastAPI app with lifespan parameter
app = FastAPI(
    title="Depth Anything V2 API",
    description="API for Depth Anything V2 depth prediction",
    version="1.0.0",
    lifespan=lifespan
)

# Model configuration
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Model used for responses
class DepthResponse(BaseModel):
    depth_array: list
    width: int
    height: int
    min_depth: float
    max_depth: float

class ModelWorker:
    def __init__(self, gpu_id: int, model_type: str = 'vitl'):
        """Initialize model worker process"""
        self.gpu_id = gpu_id
        self.model_type = model_type
        self.device = f'cuda:{gpu_id}'
        self.request_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.process = None
        
    def start(self):
        """Start model worker process"""
        self.process = mp.Process(
            target=self._worker_process, 
            args=(self.gpu_id, self.model_type, self.request_queue, self.result_queue)
        )
        self.process.daemon = True
        self.process.start()
        
    def _worker_process(self, gpu_id: int, model_type: str, request_queue: mp.Queue, result_queue: mp.Queue):
        """Worker process function, loads model and handles requests"""
        try:
            # Set current process to use GPU
            torch.cuda.set_device(gpu_id)
            
            # Initialize model
            model = DepthAnythingV2(**model_configs[model_type])
            state_dict = torch.load(f'checkpoints/depth_anything_v2_{model_type}.pth', map_location=f"cuda:{gpu_id}")
            model.load_state_dict(state_dict)
            model = model.to(f"cuda:{gpu_id}").eval()
            
            # Set color map
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            
            print(f"Model loaded to GPU:{gpu_id}")
            
            # Add counter, clean up memory every N requests
            request_count = 0
            clean_interval = 100  # Clean up every 10 requests
            
            # Request handling loop
            while True:
                try:
                    # Get request
                    request_id, image, request_type = request_queue.get()
                    
                    # Predict depth
                    with torch.no_grad():
                        depth = model.infer_image(image[:, :, ::-1])  # RGB to BGR
                    
                    # Process result based on request type
                    if request_type == "raw":
                        # Return raw depth data
                        result = {
                            "depth": depth,
                            "min_depth": float(depth.min()),
                            "max_depth": float(depth.max())
                        }
                    elif request_type == "gray":
                        # Normalize depth map to grayscale
                        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                        depth_norm = depth_norm.astype(np.uint8)
                        result = {"depth_norm": depth_norm, "mode": "L"}
                    elif request_type == "color":
                        # Normalize and create color depth map
                        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                        depth_norm = depth_norm.astype(np.uint8)
                        colored_depth = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
                        result = {"colored_depth": colored_depth}
                        
                    result_queue.put((request_id, result))
                    
                    # Clean up memory
                    request_count += 1
                    if request_count % clean_interval == 0:
                        torch.cuda.empty_cache()

                    
                except Exception as e:
                    print(f"GPU {gpu_id} worker process error: {str(e)}")
                    result_queue.put((request_id, {"error": str(e)}))
                    
        except Exception as e:
            print(f"GPU {gpu_id} initialization failed: {str(e)}")
            
    def process_image(self, image: np.ndarray, request_type: str) -> dict:
        """Submit image processing request to worker process and wait for result"""
        request_id = f"{time.time()}_{id(image)}"
        self.request_queue.put((request_id, image, request_type))
        
        # Wait for result
        while True:
            try:
                result_id, result = self.result_queue.get(timeout=30)  # 30 seconds timeout
                if result_id == request_id:
                    return result
            except queue.Empty:
                raise TimeoutError("Request processing timed out")
                
    def stop(self):
        """Stop worker process"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()

def get_next_worker():
    """Simple round-robin scheduling to get the next worker process"""
    global worker_index, worker_pool
    
    if not worker_pool:
        raise RuntimeError("No available model worker processes")
    
    worker = worker_pool[worker_index]
    worker_index = (worker_index + 1) % len(worker_pool)
    return worker

@app.post("/predict/color_depth")
async def predict_color_depth(file: UploadFile = File(...)):
    """
    Upload an image and return the color depth prediction result
    
    Parameters:
    - file: The uploaded image file
    
    Returns:
    - Color depth map image
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    try:
        # Read uploaded image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        # Convert to RGB to ensure correct format
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        # Convert to numpy array
        image = np.array(pil_image)
        
        # Get worker and process image
        worker = get_next_worker()
        result = worker.process_image(image, request_type="color")
        
        if "error" in result:
            raise Exception(result["error"])
        
        colored_depth = result["colored_depth"]
        colored_depth_pil = Image.fromarray(colored_depth)
        
        # Save and return depth map
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            colored_depth_pil.save(tmp.name)
            return FileResponse(tmp.name, media_type="image/png", filename="depth_map.png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/raw_depth_array")
async def predict_raw_array(file: UploadFile = File(...)):
    """
    Upload an image and return the raw depth data array
    
    Parameters:
    - file: The uploaded image file
    
    Returns:
    - Depth data array and its metadata
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    try:
        # Read uploaded image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        image = np.array(pil_image)
        
        # Get worker and process image
        worker = get_next_worker()
        result = worker.process_image(image, request_type="raw")
        
        if "error" in result:
            raise Exception(result["error"])
            
        depth = result["depth"]
        min_depth = result["min_depth"]
        max_depth = result["max_depth"]
        
        # Convert to list for JSON serialization
        depth_list = depth.flatten().tolist()
        
        # Return depth array data
        return DepthResponse(
            depth_array=depth_list,
            width=depth.shape[1],  
            height=depth.shape[0],
            min_depth=min_depth,
            max_depth=max_depth
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/depth")
async def predict_depth_map(file: UploadFile = File(...)):
    """
    Upload an image and return the depth prediction result
    
    Parameters:
    - file: The uploaded image file
    
    Returns:
    - Grayscale depth map image
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        image = np.array(pil_image)
        
        worker = get_next_worker()
        result = worker.process_image(image, request_type="gray")
        
        if "error" in result:
            raise Exception(result["error"])
            
        depth_norm = result["depth_norm"]
        mode = result["mode"]
        
        depth_image = Image.fromarray(depth_norm, mode=mode)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            depth_image.save(tmp.name)
            return FileResponse(tmp.name, media_type="image/png", filename="depth_map_gray.png")
        
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