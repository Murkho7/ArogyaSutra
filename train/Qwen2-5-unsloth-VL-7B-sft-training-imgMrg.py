Ddir="gpt-4o-mini/MedQA"
output_dir="Qwen25_mlang8_ActCri_sft"
model_id="Qwen/Qwen2.5-VL-7B-Instruct"


import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import glob
import re

t_dataset=[]
for file_path in glob.glob(f"{Ddir}/*.jsonl"):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t_dataset.append(json.loads(line))

                
def extract_boxed_answer(text: str) -> str:
        """
        Extract option letter from <answer>...</answer>.
        Returns 'A'–'E' or None.
        """
        if not text:
            return None
    
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            text,
            re.DOTALL | re.IGNORECASE
        )
        search_text = answer_match.group(1) if answer_match else text
    
        boxed = re.search(r"\\boxed\{([A-E])\}", search_text)
        return boxed.group(1) if boxed else None


train_dataset = []
#i=10
total=0
count=0
for item in t_dataset:
    total +=  1
    if isinstance(item["response"], list):
        response=item['response'][0]["text"]
    else:
        response=item['response']
   
    if item['answer']==extract_boxed_answer(response):
        count +=  1
        train_dataset.append(
        {
            "lid" : item["lid"],
            "language" : item["lid"].split("~")[0],
            "correct" : item["answer"],
            "messages": item["conversation"]
        })
    
print(f"After filter Dataset size: {count}")
print(f"Before filter Dataset size: {total}")



import math

# ----------------------------
# Image merge (adaptive layout)
# ----------------------------

from PIL import Image, ImageDraw, ImageFont

def merge_pil_images(images, canvas_size=(512, 512), padding=6):
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

sys_message="""You are a specialized medical multimodal Multilingual agent. Your purpose is to solve visual question answering tasks in different languages by thinking step-by-step and using tools, SHORT_MEMORY and LONG_MEMORY. You must give your final correct option from A to E within <FINAL_ANSWER>A/B/C/D/E</FINAL_ANSWER> tag in the end.

# Tools 
You are provided with following tools: 
1. object_detection: Detects objects in an image. parameters: image_id, objects (list of object names). 
2. zoom_in: Zooms in on a specified bounding box in an image. parameters: image_id, bbox (bounding box coordinates), factor (zoom factor). 
3. edge_detection: Detects edges in an image. parameters: image_id. 
4. depth_estimation: Estimates depth in an image. parameters: image_id. 

# Instruction 
1. At first you need to detect the language of the question. All your answer would be in the detected language.
2. In each turn, you should start with <think> tag. In this tag, you need to conduct a step-by-step reasoning process about the image and question and evaluatethe tool usage and give the reason. Based on received tool results, you also need to analyze them. You need to use a minimum of 1 tool call every time to get the correct response. 
3. If you think some tools are useful, call them in <tool_call> tag.
4. After that you can answer in <answer> tag after each trial. You need to put a simple and direct answer in \\boxed{}.
5. If you want to rethink, Use <SHORT_MEMORY> and <LONG_MEMORY> to gathar previous trial informations before arriving to the final answer.
6. After all the trails when you are ready to give the final answer put the correct option inside <FINAL_ANSWER></FINAL_ANSWER> tag. 
7. After each <think> tag use <answer> tag. After all the <think> and <answer> tag use only ONE <FINAL_ANSWER></FINAL_ANSWER> tag containing your last final answer. So that it can be extracted for verification. """


for item in train_dataset:
    for m in item["messages"]:
        if m["role"] == "system" :
            m["content"][0]["text"]=sys_message

results_out = []
import re
for rec in train_dataset:
    new_rec=rec
    msg=[]
    mergeMsg=""
    conv1= rec["messages"]
    st=0
    for m in conv1:
        if m["role"] == "assistant" : st=1
        if st==1:
            for item in m["content"]:
                if item.get("type") == "text":
                    #print("jhdv")
                    mergeMsg += "\n\n" + re.sub(r"\s+", " ", re.sub(r"\n+", "\n", item.get("text", ""))).strip()
    if "<FINAL_ANSWER>" not in  mergeMsg:
        mergeMsg+= f"\n<FINAL_ANSWER>{rec['correct']}</FINAL_ANSWER>"
    msg.append(rec["messages"][0])
    msg.append(rec["messages"][1])
    msg.append({'role': 'assistant','content': [{'type': 'text', 'text': mergeMsg}]})
    new_rec["messages"]=msg
    results_out.append(new_rec)
    
    

from PIL import Image
from typing import List, Dict, Any
import copy

def convert_dataset_with_images(
    dataset: List[Dict[str, Any]],
    convert_to_rgb: bool = True,
) -> List[Dict[str, Any]]:
    """
    Return a NEW dataset with:
      - image_url blocks replaced by {type: 'image', image: PIL.Image}
      - images stored in sample['images']
      - original dataset left untouched
    """

    new_dataset = []

    for sample in dataset:
        sample_copy = copy.deepcopy(sample)

        loaded_images = []
        image_cache = {}

        for msg in sample_copy.get("messages", []):
            if msg.get("role") != "user":
                continue

            new_content = []

            for block in msg.get("content", []):
                if block.get("type") == "text":
                    if "Analyze the medical images and answer the correct option from multiple-choice question" in block["text"]:
                        block["text"]=block["text"].replace("Figure A:","Figure :")
                        block1=block
                        continue
                    if "Question:" in block["text"]:
                        block3=block
                                 
                if block.get("type") == "image_url":
                    path = block["image_url"]["url"]
                    #full_path = path.partition("/")[2]
                    full_path=path

                    if full_path not in image_cache:
                        img = Image.open(full_path)
                        if convert_to_rgb:
                            img = img.convert("RGB")
                        image_cache[full_path] = img

                    img = image_cache[full_path]
                    loaded_images.append(img)
            #print(f"{sample['cid']}--{len(loaded_images)}")
            new_content.append(block1)
            new_content.append({
                        "type": "image",
                        "image": merge_pil_images(loaded_images),
                    })
            new_content.append(block3)
            msg["content"] = new_content

        sample_copy["image"] = merge_pil_images(loaded_images)
        new_dataset.append(sample_copy)

    return new_dataset

#print(convert_dataset_with_images(results_out))    

training_records=convert_dataset_with_images(results_out)

import random
rng = random.Random(42)
rng.shuffle(training_records)

split_idx = int(len(training_records) * 0.95)
train_dataset = training_records[:split_idx]
eval_dataset = training_records[split_idx:]
print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Eval Dataset Size: {len(eval_dataset)}")
print(training_records[1])



from huggingface_hub import login
api_token = "YOUR_HF_TOKEN"
login(token=api_token)


from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    model_id,
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    
)


model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


from transformers.trainer_utils import get_last_checkpoint
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


FastVisionModel.for_training(model) # Enable for training!
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer, max_seq_length=50000, resize="max"), # Must use!
    train_dataset = train_dataset,
    #eval_dataset=eval_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        #max_steps = 4410,
        num_train_epochs = 2, # Set this instead of max_steps for full training runs
        learning_rate = 5e-5,
        logging_steps = 100,
        #eval_steps=50,  # Steps interval for evaluation
        #eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
    	save_steps=210,
        max_grad_norm=1.0,
    	optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",     # For Weights and Biases
        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        max_seq_length=50000,
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length=50000,   # or set a very high value
        output_dir=f"{output_dir}",  # Directory to save the model
        logging_dir=f"{output_dir}_log",
        save_total_limit=2,
        logging_first_step=True
    	#push_to_hub=True,
    ),
) 



import os
last_checkpoint = None
if os.path.isdir(f"{output_dir}"):
    last_checkpoint = get_last_checkpoint(f"{output_dir}")

if last_checkpoint is not None:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("No checkpoint found, starting fresh training...")
    trainer.train()

trainer.save_model(f"{output_dir}")
tokenizer.save_pretrained(f"{output_dir}")

model.save_pretrained_merged(
    f"{output_dir}-merged",
    tokenizer,
    save_method = "merged_16bit_forced",  # or "merged_4bit"
)

from huggingface_hub import create_repo, upload_folder

create_repo(
    repo_id="iit-patna-cse-ai/arogyasutraV1E3.5-t",
    repo_type="model",
    exist_ok=True,
)

upload_folder(
    folder_path= f"{output_dir}-merged",
    repo_id="iit-patna-cse-ai/arogyasutraV1E3.5-t",
    repo_type="model",
)
