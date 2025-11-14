#!/bin/bash
#SBATCH --job-name=infer_llama_32_3b
#SBATCH --output=infer_output/infer_llama_32_3b_%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G


# Activate your environment
source /apps/local/anaconda3/conda_init.sh
conda activate llm

# Run your script
python -m src.model.inferenec_batch --split test --model_name blahblahblah --savefolder infered --batch_size 8