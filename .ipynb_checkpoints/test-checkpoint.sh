#!/bin/bash
#lol=["English", "Hindi", "Bengali", "Assamese" ,"Telugu", "Marathi",  "Tamil", "Punjabi"]


module load MLDL/miniconda3
module load cuda/12.8
conda activate asenv


python -u test/agent_MedQA_eval_test_sft_imgMrg.py
