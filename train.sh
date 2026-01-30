#!/bin/bash

module load MLDL/miniconda3
conda activate tools

python train/Qwen2-5-unsloth-VL-7B-sft-training-imgMrg.py
