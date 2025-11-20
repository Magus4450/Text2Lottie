#!/bin/bash
#SBATCH --job-name=infer_llama_32_3b_tokenized_base
#SBATCH --output=infer_output/infer_llama_32_3b_tokenized_base_%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G


# Activate your environment
source /apps/local/anaconda3/conda_init.sh
conda activate finetuen

# Run your script
python -m src.model.tokenized.inference_tokenized_test_set