#!/bin/bash
<<<<<<< HEAD
#SBATCH --job-name=finetune_llama_32_3b_tokenized_aug
#SBATCH --output=finetune_output/finetune_llama_32_3b_tokenized_aug_%j.out
=======
#SBATCH --job-name=finetune_llama_32_3b_base
#SBATCH --output=finetune_output/finetune_llama_32_3b_base_%j.out
>>>>>>> 59d43c4 (final train)
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G


# Activate your environment
source /apps/local/anaconda3/conda_init.sh
conda activate llm

# Run your script
<<<<<<< HEAD
python -m src.model.tokenized.train_tokenized
=======
python -m src.model.base.train_base
>>>>>>> 59d43c4 (final train)
