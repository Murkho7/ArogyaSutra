#!/bin/bash

module load MLDL/miniconda3
module load cuda/12.8

conda activate asenv

python -u gen/agent_MedQA_eval_gpt.py