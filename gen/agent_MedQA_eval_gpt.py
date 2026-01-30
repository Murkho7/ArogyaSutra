import sys
dataset="iit-patna-cse-ai/ArogyaBodha_ActCri"
api_token = "YOUR_HF_TOKEN"  
split="train"
Lang=""
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
from PIL import Image
from omegaconf import DictConfig
from datasets import load_dataset, Dataset
from typing import List, Tuple, Dict, Any, Optional, Callable
from prompt import *
from verl.workers.agent import APIAgentAC


os.environ["OPENAI_API_KEY"] ="YOUR_OPEN_AI_TOKEN"


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
                "max_turns": 3,
                "max_tokens_per_turn": 4096,
                "save_intermediate_responses": True,
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "openai_model": self.model_name,
                "openai_temperature": 0.5,
            })
        else:
            agent_config = DictConfig({
                "max_turns": 3,
                "max_tokens_per_turn": 4096,
                "save_intermediate_responses": True,
                "openai_api_base": f"http://localhost:{port}/v1/chat/completions",
                "openai_api_key": "fake-api-key",
                "openai_model": self.model_name,
                "openai_temperature": 0.2,
            })
        
        return APIAgentAC(agent_config)

    def _base_process_sample(self, idx: int, prompt: str, images: List[Image.Image], 
                           image_dir: str, output_path: str, result_data: Dict[str, Any]) -> Tuple[int, bool]:
        """Base sample processing function that contains common logic"""
        try:
            port = self.get_next_port()
            
            agent = self.create_agent(port)
            response, conversation = agent.chat_with_tools(
                system_prompt=self.instruction_following.replace("<<LANG>>",result_data["language"]),
                prompt=prompt,
                images=images,
                gtl={"q":result_data["question"],
                     "o":result_data["options"],
                     "a":result_data["answer"],
                     "eq":result_data["Eng_Q"],
                     "eo":result_data["Eng_O"],
                     "l":result_data["language"]}
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
        
        sample_images = []
        image_placeholder="Images: "
        
        #Handle images - may have multiple images
        ##sample_images=[img for img in sample.values() if isinstance(img, Image.Image)]
        
        for i in range(1, 5):  #  can have up to 5 images
            img_key = f'Figure_{chr(64+i)}'
            if img_key in sample and sample[img_key] is not None:
                image_placeholder += f'Figure {chr(64+i)}: <image> '
                sample_images.append(sample[img_key])
        #print("sample_images")
        print(len(sample_images))        
        if len(sample_images) ==1:
            image_placeholder = image_placeholder.replace("A","")
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
            from huggingface_hub import login
            login(token=api_token)
            
            ds = load_dataset(dataset)
            return ds[split].select(range(2)).map(self.resize_images)

##################################################### return ds["test"].select(range(2))
                
        
        self._base_evaluate("MedQA", load_MedQA, self.process_MedQA_sample)


def get_system_prompt(prompt_type: str) -> str:
    """Get system prompt"""
    # Get system prompt
    if prompt_type == 'agent_api':
        system_prompt = SYN_PROMPT #########rectify lang
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return system_prompt


def main():
    parser = argparse.ArgumentParser(description='Run dataset evaluation')
    #parser.add_argument('--model-name', type=str, default='qwen', help='Model name') #-->
    parser.add_argument('--model-name', type=str, default='gpt-4o-mini', help='Model name') #-->
    parser.add_argument('--port-pool', type=str, default='10010', help='API server port pool, comma separated')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel worker threads')
    parser.add_argument('--dataset', type=str, default='MedQA', help='Dataset to evaluate')
    parser.add_argument('--prompt', type=str, default='agent_api', help='Prompt type')
    #parser.add_argument('--remote',type=bool , default=False, help='Use remote API') #-->
    parser.add_argument('--remote',type=bool , default=True, help='Use remote API') #-->
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
