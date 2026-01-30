## beofre testing starting qwen server is mandatory

dataset=f"iit-patna-cse-ai/ArogyaBodha_ActCri"
split="test"


import os
import json
import time
import base64
import argparse
from tqdm import tqdm
import threading
import concurrent.futures
import signal
import sys
import re
from PIL import Image, ImageDraw, ImageFont
from omegaconf import DictConfig
from datasets import load_dataset, Dataset
from typing import List, Tuple, Dict, Any, Optional, Callable
from prompt import *
from verl.workers.agent import APIAgentsft
import math


class DatasetEvaluator:
    def __init__(self, model_name: str, port_pool: List[int], isremote: bool, workers: int, instruction_following: str):
        self.model_name = model_name
        self.isremote = isremote
        self.port_pool = port_pool
        self.workers = workers
        self.instruction_following = instruction_following
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Port round-robin related
        self.port_lock = threading.Lock()
        self.port_index = 0
        
        # File write lock
        self.file_lock = threading.Lock()
        
        # Thread pool executor
        self.executor = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\nGracefully shutting down, please wait...")
        if self.executor is not None:
            self.executor.shutdown(wait=False)
        sys.exit(0)

    def get_next_port(self) -> int:
        """Get the next available port from the pool in round-robin"""
        with self.port_lock:
            current_port = self.port_pool[self.port_index]
            self.port_index = (self.port_index + 1) % len(self.port_pool)
            return current_port

    def save_base64_image(self, base64_str: str, image_dir: str, sample_id: int, image_index: int) -> str:
        """Process base64 image and save to file"""
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]
        
        image_data = base64.b64decode(base64_str)
        image_format = "png"
        filename = f"{sample_id}_{image_index}.{image_format}"
        filepath = os.path.join(image_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        return filename
        
    def resize_images(self, example):
        """
        Resize all images in the example to max 512x512 if they're larger.
        Maintains aspect ratio and only resizes if image exceeds 512 in either dimension.
        """
        max_size = 512
        
        # Find all image columns (figure_a, figure_b, image, etc.)
        for key, value in example.items():
            # Check if the value is a PIL Image
            if isinstance(value, Image.Image):
                width, height = value.size
                
                # Only resize if image is larger than max_size in either dimension
                if width > max_size or height > max_size:
                    # Calculate new dimensions maintaining aspect ratio
                    if width > height:
                        new_width = max_size
                        new_height = int(height * (max_size / width))
                    else:
                        new_height = max_size
                        new_width = int(width * (max_size / height))
                    
                    # Resize image with high-quality resampling
                    example[key] = value.resize((new_width, new_height), Image.LANCZOS)
        
        return example


    def process_conversation_images(self, conversation: List[Dict], image_dir: str, sample_id: int) -> List[Dict]:
        """Process images within the conversation messages"""
        new_conversation = []
        image_index = 0
        
        for message in conversation:
            new_message = message.copy()
            
            if isinstance(message.get('content'), list):
                new_content = []
                for item in message['content']:
                    if item.get('type') == 'image_url' and 'image_url' in item:
                        url = item['image_url'].get('url', '')
                        if url.startswith('data:image'):
                            filename = self.save_base64_image(url, image_dir, sample_id, image_index)
                            image_index += 1
                            new_content.append({
                                'type': 'image_url',
                                'image_url': {'url': f"{image_dir}/{filename}"}
                            })
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)
                new_message['content'] = new_content
            
            new_conversation.append(new_message)
        
        return new_conversation

    def create_agent(self, port: int):
        """Create an Agent instance"""
        if self.isremote:
            print("Remote Agent GPT set as default.")
            agent_config = DictConfig({
                "max_turns": 1,
                "max_tokens_per_turn": 4096,
                "save_intermediate_responses": True,
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "openai_model": self.model_name,
                "openai_temperature": 0.5,
            })
        else:
            agent_config = DictConfig({
                "max_turns": 1,
                "max_tokens_per_turn": 6096,
                "save_intermediate_responses": True,
                "openai_api_base": f"http://localhost:{port}/v1/chat/completions",
                "openai_api_key": "fake-api-key",
                "openai_model": self.model_name,
                "openai_temperature": 0.0,
                "openai_top_p" : 1.00
            })
        
        return APIAgentsft(agent_config)


# ----------------------------
# Image merge (adaptive layout)
# ----------------------------

    def merge_pil_images(self, images, canvas_size=(512, 512), padding=6):
        n = len(images)
        W, H = canvas_size
        canvas = Image.new("RGB", (W, H), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
    
        draw_labels = n > 1
    
        def resize_fit(img, w, h):
            r = min(w / img.width, h / img.height)
            return img.resize((int(img.width * r), int(img.height * r)))
    
        def draw_label(x, y, text):
            pad = 4
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            draw.rectangle([x, y, x+tw+2*pad, y+th+2*pad], fill="black")
            draw.text((x+pad, y+pad), text, fill="white", font=font)
    
        # ---------- SINGLE IMAGE ----------
        if n == 1:
            img = resize_fit(images[0], W-2*padding, H-2*padding)
            canvas.paste(img, ((W-img.width)//2, (H-img.height)//2))
            return canvas
    
        # ---------- TWO IMAGES (ASPECT-AWARE) ----------
        if n == 2:
            img0 = images[0]
    
            # Decide layout using first image aspect ratio
            portrait = img0.height > img0.width
    
            if portrait:
                # SIDE-BY-SIDE
                cell_w = (W - 3*padding) // 2
                cell_h = H - 2*padding
    
                for i, img in enumerate(images):
                    img = resize_fit(img, cell_w, cell_h)
                    x = padding + i*(cell_w + padding) + (cell_w-img.width)//2
                    y = (H-img.height)//2
                    canvas.paste(img, (x, y))
                    if draw_labels:
                        draw_label(x+5, y+5, f"Figure {chr(65+i)}")
    
            else:
                # TOP-DOWN
                cell_h = (H - 3*padding) // 2
                cell_w = W - 2*padding
    
                for i, img in enumerate(images):
                    img = resize_fit(img, cell_w, cell_h)
                    x = (W-img.width)//2
                    y = padding + i*(cell_h + padding) + (cell_h-img.height)//2
                    canvas.paste(img, (x, y))
                    if draw_labels:
                        draw_label(x+5, y+5, f"Figure {chr(65+i)}")
    
            return canvas
    
        # ---------- FALLBACK (3+ images, grid) ----------
        import math
        cols = math.ceil(n ** 0.5)
        rows = math.ceil(n / cols)
        cw = (W - (cols+1)*padding)//cols
        ch = (H - (rows+1)*padding)//rows
    
        for i, img in enumerate(images):
            r, c = divmod(i, cols)
            img = resize_fit(img, cw, ch)
            x = padding + c*(cw+padding) + (cw-img.width)//2
            y = padding + r*(ch+padding) + (ch-img.height)//2
            canvas.paste(img, (x, y))
            if draw_labels:
                draw_label(x+5, y+5, f"Figure {chr(65+i)}")
    
        return canvas
    def _base_process_sample(self, idx: int, prompt: str, images: List[Image.Image], 
                           image_dir: str, output_path: str, result_data: Dict[str, Any]) -> Tuple[int, bool]:
        """Base sample processing function that contains common logic"""
        try:
            port = self.get_next_port()
            
            agent = self.create_agent(port)
            response, conversation = agent.chat_with_tools(
                system_prompt=self.instruction_following,
                prompt=prompt,
                images=images,
                #gt=result_data["answer"]
            )
            
            conversation = self.process_conversation_images(conversation, image_dir, idx)
            
            # Build base result
            result = {
                'sample_id': idx,
                'response': response,
                'conversation': conversation,
                **result_data  # merge provided result data
            }
            
            
            # Write to file
            with self.file_lock:
                with open(output_path, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            
            return idx, True
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return idx, False

    def process_MedQA_sample(self, args: Tuple) -> Tuple[int, bool]:
        """Process MedQA sample"""
        idx, sample, image_dir, output_path = args
        
        # Extract sample data
        q = f"\n Question: {sample['Question']} \n"
        o = str(sample['Options'])
        a = sample['Correct_answer']
        
        
        image_placeholder="Images: <image>"
        
        #Handle images - may have multiple images
        sample_images=[self.merge_pil_images([img for img in sample.values() if isinstance(img, Image.Image)])]
        
        # for i in range(1, 5):  #  can have up to 5 images
        #     img_key = f'Figure {chr(64+i)}'
        #     if img_key in sample and sample[img_key] is not None:
        #         image_placeholder += f'Figure {chr(64+i)}: <image> '
        #         sample_images.append(sample[img_key])
        # #print("sample_images")
        # print(len(sample_images))        
        # if len(sample_images) ==1:
        #     image_placeholder = image_placeholder.replace("A","")
        #image_placeholder = '<image>' * len(sample_images)
        

        
        # Build prompt with options
        prompt = "Analyze the medical images and answer the correct option from multiple-choice question: " + image_placeholder + q + "\nAnswer Choices:" + o


        
        result_data = {
            'split' : split,
            'lid' : sample['lid'],
            'question': q,
            'options': o,
            'answer': a,
            'language' : sample['language'],
            'source' : sample['source'],
            'Eng_Q' : sample['Question_Eng'],
            'Eng_O' : sample['Options_Eng']
            
        }
        
        return self._base_process_sample(idx, prompt, sample_images, image_dir, output_path, result_data)


    def run_parallel_evaluation(self, args_list: List[Tuple], process_func: Callable) -> Tuple[int, int]:
        """Run parallel evaluation"""
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
        try:
            results = list(tqdm(self.executor.map(process_func, args_list), total=len(args_list)))
        finally:
            self.executor.shutdown()
            self.executor = None
        
        success_count = sum(1 for _, success in results if success)
        fail_count = len(results) - success_count
        return success_count, fail_count

    def _base_evaluate(self, dataset_name: str, dataset_loader: Callable, process_func: Callable, 
                      dataset_args: Optional[Dict] = None):
        """Base evaluation function that contains common evaluation flow"""
        # Set hierarchical output paths: evaluation/model/benchmark
        base_dir = f"{self.model_name}/{dataset_name}"
        output_path = f"{base_dir}/results_{self.timestamp}.jsonl"
        image_dir = f"{base_dir}/images_{self.timestamp}"
        
        # Create directories
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        
        # Create empty file if not exists
        if not os.path.exists(output_path):
            with open(output_path, 'w') as f:
                pass
        
        # Load dataset
        print(f"Loading {dataset_name} dataset...")
        if dataset_args:
            dataset = dataset_loader(**dataset_args)
        else:
            dataset = dataset_loader()
        
        print(f"Starting evaluation of {len(dataset)} {dataset_name} samples using {self.workers} parallel worker threads")
        
        # Prepare argument list
        args_list = [(idx, sample, image_dir, output_path) 
                    for idx, sample in enumerate(dataset)]
        
        # Run evaluation
        success_count, fail_count = self.run_parallel_evaluation(args_list, process_func)
        
        print(f"{dataset_name} evaluation complete, results saved to {output_path}")
        print(f"Successfully processed: {success_count} samples, Failed: {fail_count} samples")

    # Evaluation methods for each dataset
    def evaluate_MedQA(self):
        def load_MedQA():
            # from huggingface_hub import login
            # api_token = "YOUR_HF_TOKEN"
            # login(token=api_token)
            #ds = load_dataset(dataset)
            
            ds = load_dataset(dataset, download_mode="reuse_cache_if_exists" )
            return ds[split].select(range(2)).map(self.resize_images)

##################################################### return ds["test"].select(range(2))
                
        
        self._base_evaluate("MedQA", load_MedQA, self.process_MedQA_sample)


def get_system_prompt(prompt_type: str) -> str:
    """Get system prompt"""
    # Get system prompt
    if prompt_type == 'agent':
        system_prompt = RL_PROMPT
    elif prompt_type == 'text':
        system_prompt = TEXT_RL_PROMPT
    elif prompt_type == 'none':
        system_prompt = None
    elif prompt_type == 'agent_api':
        system_prompt = SYN_PROMPT.replace("<<LANG>>",Lang)
    elif prompt_type == 'agentM':
        system_prompt = MEMORY_PROMPT
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return system_prompt


def main():
    parser = argparse.ArgumentParser(description='Run dataset evaluation')
    parser.add_argument('--model-name', type=str, default='qwen', help='Model name') #-->
    parser.add_argument('--port-pool', type=str, default='10010', help='API server port pool, comma separated')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel worker threads')
    parser.add_argument('--dataset', type=str, default='MedQA', help='Dataset to evaluate')
    parser.add_argument('--prompt', type=str, default='agentM', help='Prompt type') #-->
    parser.add_argument('--remote',type=bool , default=False, help='Use remote API') #-->
    # Allow unknown Jupyter arguments
    args, unknown = parser.parse_known_args()
    print("Ignoring unknown args:", unknown)
    #args = parser.parse_args()

    # Parse port pool
    port_pool = [int(port.strip()) for port in args.port_pool.split(',')]
    print(f"Using port pool: {port_pool}")
    
    # Get system prompt
    try:
        instruction_following = get_system_prompt(args.prompt)
        print(f"System prompt: {instruction_following}")
    except ValueError as e:
        print(str(e))
        return
    
    # Create evaluator
    evaluator = DatasetEvaluator(
        model_name=args.model_name,
        port_pool=port_pool,
        isremote=args.remote,
        workers=args.workers,
        instruction_following=instruction_following,
    )
    # Mapping of dataset evaluation methods
    dataset_methods = {
        'MedQA': evaluator.evaluate_MedQA, 
    }
    
    # Run the selected evaluation method
    if args.dataset in dataset_methods:
        dataset_methods[args.dataset]()
    else:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available datasets: {', '.join(dataset_methods.keys())}")
        return


if __name__ == "__main__":
    main()
