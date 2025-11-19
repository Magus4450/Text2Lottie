create-dataset:
	python -m src.model.build_hf_instruction_dataset

train:
	sbatch train.slurm

train-tokenized:
	sbatch train_tokenized.slurm

infer:
	sbatch infer.slurm
