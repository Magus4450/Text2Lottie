#!/bin/bash
#SBATCH --job-name=finetune_llama_32_3b_base
#SBATCH --output=finetune_output/finetune_llama_32_3b_base_%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G


# Activate your environment
source /apps/local/anaconda3/conda_init.sh
conda activate llm

# Run your script
python -m src.model.base.train_base
