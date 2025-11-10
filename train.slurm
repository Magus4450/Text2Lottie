#!/bin/bash
#SBATCH --job-name=finetune_llama_3
#SBATCH --output=finetune_llama_3_%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G


# Activate your environment
source /apps/local/anaconda3/conda_init.sh
conda activate finetuen

# Run your script
python -m src.model.train_hf