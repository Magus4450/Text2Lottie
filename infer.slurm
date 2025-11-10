#!/bin/bash
#SBATCH --job-name=infer_qwen3_8b
#SBATCH --output=infer_qwen3_8b_%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G


# Activate your environment
source /apps/local/anaconda3/conda_init.sh
conda activate finetuen

# Run your script
python -m src.model.inference \
  --prompt "Generate a lottie JSON animation of a red circle moving left to right." \
  --temperature 0.6 --top_p 0.95 --max_new_tokens 800