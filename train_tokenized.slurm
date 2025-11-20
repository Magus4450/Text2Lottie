#!/bin/bash
#SBATCH --job-name=finetune_llama_32_3b_tokenized_base_wo_svg
#SBATCH --output=finetune_output/finetune_llama_32_3b_tokenized_base_wo_svg_%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G


# Activate your environment
source /apps/local/anaconda3/conda_init.sh
conda activate finetuen

# Run your script
python -m src.model.tokenized.train_tokenized