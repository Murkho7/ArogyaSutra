#!/bin/bash

module load MLDL/miniconda3
module load cuda/12.8
conda activate asenv

bash -u tservm.sh
