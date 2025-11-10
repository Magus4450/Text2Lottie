# src/model/train_full.py
import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from src.model.semantic_tokenizer import LottieSemanticTokenizer, to_semantic
import src.model.config as config

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("No CUDA device detected.")


# -----------------------------
# Load base model and tokenizer
# -----------------------------
print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    # torch_dtype=config.DTYPE,
    device_map="auto",
)
print(model.dtype, next(model.parameters()).device)

# -----------------------------
# Augment tokenizer with Lottie semantic tags
# -----------------------------
print("Augmenting tokenizer with Lottie semantic tags/patterns...")
_ = LottieSemanticTokenizer(tokenizer, add_as_special_tokens=False)

old_vocab = model.get_input_embeddings().weight.size(0)
new_vocab = len(tokenizer)
print(f"Resizing embeddings: {old_vocab} → {new_vocab}")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

# -----------------------------
# Ensure tokenizer has a pad token
# -----------------------------
if tokenizer.pad_token is None:
    print("Tokenizer has no pad_token; setting pad_token = eos_token.")
    tokenizer.pad_token = tokenizer.eos_token

# Ensure embeddings are trainable
model.get_input_embeddings().requires_grad_(True)
model.get_output_embeddings().requires_grad_(True)

model.gradient_checkpointing_enable()

# -----------------------------
# Load train/val/test datasets
# -----------------------------
DATASET_TRAIN = getattr(config, "DATASET_TRAIN", "train.jsonl")
DATASET_VAL = getattr(config, "DATASET_VAL", "val.jsonl")
DATASET_TEST = getattr(config, "DATASET_TEST", "test.jsonl")
NUM_PROC = getattr(config, "DATASET_NUM_PROC", 4)

data_files = {}
if os.path.exists(DATASET_TRAIN):
    data_files["train"] = DATASET_TRAIN
if os.path.exists(DATASET_VAL):
    data_files["validation"] = DATASET_VAL
if os.path.exists(DATASET_TEST):
    data_files["test"] = DATASET_TEST

if not data_files:
    raise FileNotFoundError("No dataset files found. Expected train/val/test JSONL paths in config.")

print(f"Loading dataset from files: {data_files}")
dataset_splits = load_dataset("json", data_files=data_files)

# -----------------------------
# Filter malformed rows and apply semantic conversion
# -----------------------------
def apply_chat_template(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    text = to_semantic(text)
    return {"text": text}

for split in list(dataset_splits.keys()):
    dataset_splits[split] = dataset_splits[split].filter(
        lambda ex: isinstance(ex.get("messages", None), list) and len(ex["messages"]) >= 2
    )
    dataset_splits[split] = dataset_splits[split].map(
        apply_chat_template,
        num_proc=NUM_PROC,
    )

print({k: len(v) for k, v in dataset_splits.items()})
print("Example formatted text:\n", dataset_splits["train"][0]["text"])


# -----------------------------
# Tokenization
# -----------------------------
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=config.MAX_SEQ_LENGTH,
    )

tokenized = dataset_splits.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset_splits["train"].column_names,
    num_proc=NUM_PROC,
)

# -----------------------------
# Inspect tokenized sample
# -----------------------------
print("\nInspecting one tokenized example...")
sample = tokenized["train"][0]
print("Token IDs:", sample["input_ids"][:100])  # first 50 token IDs
print("Decoded text:", tokenizer.decode(sample["input_ids"])[:100])

# for tok in sample["input_ids"][:100]:
#     print(tok, tokenizer.decode(tok))

# -----------------------------
# Data collator
# -----------------------------
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -----------------------------
# Training arguments
# -----------------------------
print("Setting up training...")
training_args = TrainingArguments(
    output_dir=config.OUTPUT_DIR,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
    learning_rate=config.LEARNING_RATE,
    warmup_steps=config.WARMUP_STEPS,
    num_train_epochs=config.NUM_EPOCHS,
    weight_decay=config.WEIGHT_DECAY,
    logging_steps=config.LOGGING_STEPS,
    save_strategy=config.SAVE_STRATEGY,
    save_total_limit=config.SAVE_TOTAL_LIMIT,
    eval_strategy=config.EVAL_STRATEGY,
    load_best_model_at_end=config.LOAD_BEST_MODEL,
    lr_scheduler_type=config.LR_SCHEDULER,
    optim=config.OPTIMIZER,
    fp16=True,
    bf16=False,
    seed=config.SHUFFLE_SEED,
    report_to=config.REPORT_TO,
    gradient_checkpointing=getattr(config, "USE_GRADIENT_CHECKPOINTING", False),
    use_cache=False
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized.get("validation"),
    tokenizer=tokenizer,
    data_collator=collator,
)

# -----------------------------
# Train
# -----------------------------
print("\nStarting full fine-tuning...")
train_result = trainer.train()

print("\nSaving final model and tokenizer...")
trainer.save_model(config.MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(config.MODEL_OUTPUT_DIR)

print("Training finished!")
print(train_result)

if "test" in tokenized:
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(tokenized["test"])
    print(metrics)
