# src/model/train.py
import os
import torch
import src.model.config as config
from datasets import load_dataset, DatasetDict
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer

# Import the semantic tokenizer utilities
from src.model.semantic_tokenizer import LottieSemanticTokenizer, to_semantic

# -------------------------
# Config fallbacks (edit in config.py if you like)
# -------------------------
DATASET_JSONL = getattr(config, "DATASET_JSONL", "instruction_dataset.jsonl")
DATA_FILES = getattr(config, "DATA_FILES", None)  # e.g., {"train":"train.jsonl","validation":"val.jsonl"}
VAL_SPLIT = getattr(config, "VAL_SPLIT", 0.02)
TEST_SPLIT = getattr(config, "TEST_SPLIT", 0.02)
DATASET_NUM_PROC = getattr(config, "DATASET_NUM_PROC", 4)

# Print configuration
config.print_config()

# -------------------------
# Load model and tokenizer
# -------------------------
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.MODEL_NAME,
    max_seq_length=config.MAX_SEQ_LENGTH,
    dtype=config.DTYPE,
    load_in_4bit=config.LOAD_IN_4BIT,
)

# Unsloth/causal LMs usually prefer right padding
try:
    tokenizer.padding_side = "right"
except Exception:
    pass

# -------------------------
# Lottie semantic tokenizer (adds tokens + provides to_semantic())
# -------------------------
print("Augmenting tokenizer with Lottie semantic tags/patterns...")
# This adds <...> key tags and common literal patterns (e.g. '"a":0') to vocab
_ = LottieSemanticTokenizer(tokenizer, add_as_special_tokens=False)

# Ensure embeddings match new vocab size
try:
    model.resize_token_embeddings(len(tokenizer))
except Exception:
    # Some Unsloth models manage this internally; safe to ignore if unsupported
    pass

# -------------------------
# Add LoRA adapters
# -------------------------
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=config.LORA_R,
    target_modules=config.TARGET_MODULES,
    lora_alpha=config.LORA_ALPHA,
    lora_dropout=config.LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
    random_state=config.SHUFFLE_SEED,
    use_rslora=False,
    loftq_config=None,
)

# -------------------------
# Data preparation (JSONL with chat `messages`)
# -------------------------
def _load_chat_dataset():
    """
    Loads JSON/JSONL with rows like:
      {"id": "...",
       "messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}],
       "metadata": {...}}
    If DATA_FILES is provided in config, use it; else read single JSONL and split.
    """
    if DATA_FILES and isinstance(DATA_FILES, dict):
        print(f"Loading dataset from files: {DATA_FILES}")
        ds = load_dataset("json", data_files=DATA_FILES)
        if "train" not in ds:
            raise ValueError("DATA_FILES must include at least a 'train' split.")
        dd = DatasetDict()
        dd["train"] = ds["train"]
        if "validation" in ds:
            dd["validation"] = ds["validation"]
        else:
            split = dd["train"].train_test_split(test_size=VAL_SPLIT, seed=config.SHUFFLE_SEED)
            dd["train"], dd["validation"] = split["train"], split["test"]
        if "test" in ds:
            dd["test"] = ds["test"]
        return dd
    else:
        if not os.path.exists(DATASET_JSONL):
            raise FileNotFoundError(f"Could not find dataset file: {DATASET_JSONL}")
        print(f"Loading dataset from: {DATASET_JSONL}")
        ds = load_dataset("json", data_files={"train": DATASET_JSONL})["train"]
        # Create val/test splits from a single file
        if TEST_SPLIT and TEST_SPLIT > 0:
            tmp = ds.train_test_split(test_size=TEST_SPLIT, seed=config.SHUFFLE_SEED)
            train_val = tmp["train"]
            test = tmp["test"]
        else:
            train_val = ds
            test = None
        if VAL_SPLIT and VAL_SPLIT > 0:
            split = train_val.train_test_split(test_size=VAL_SPLIT, seed=config.SHUFFLE_SEED)
            train, val = split["train"], split["test"]
        else:
            train, val = train_val, None

        dd = DatasetDict(train=train)
        if val is not None:
            dd["validation"] = val
        if test is not None:
            dd["test"] = test
        return dd

print("Preparing dataset...")
dataset_splits = _load_chat_dataset()
print(dataset_splits)

# (Optional) filter out malformed rows
for split in list(dataset_splits.keys()):
    dataset_splits[split] = dataset_splits[split].filter(
        lambda ex: isinstance(ex.get("messages", None), list) and len(ex["messages"]) >= 2
    )

# -------------------------
# Map to 'text' using chat template + Lottie semantic encoding
# -------------------------
def _apply_chat_template(example):
    # 1) Apply the model’s chat template to get a single training string
    #    Important: add_generation_prompt=False so only assistant texts are targets.
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    # 2) Convert only fenced ```json / ```lottie blocks to semantic tags
    text = to_semantic(text)
    return {"text": text}

for split in list(dataset_splits.keys()):
    dataset_splits[split] = dataset_splits[split].map(
        _apply_chat_template,
        num_proc=DATASET_NUM_PROC,
    )

print({k: len(v) for k, v in dataset_splits.items()})
print("Example formatted text:", dataset_splits["train"][0]["text"][:200].replace("\n", " ") + " ...")

# -------------------------
# Training arguments
# -------------------------
print("\nSetting up training...")
training_args = TrainingArguments(
    per_device_train_batch_size=config.BATCH_SIZE,
    gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=config.WARMUP_STEPS,
    num_train_epochs=config.NUM_EPOCHS,
    learning_rate=config.LEARNING_RATE,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=config.LOGGING_STEPS,
    optim=config.OPTIMIZER,
    weight_decay=config.WEIGHT_DECAY,
    lr_scheduler_type=config.LR_SCHEDULER,
    seed=config.SHUFFLE_SEED,
    output_dir=config.OUTPUT_DIR,
    report_to=config.REPORT_TO,
    save_strategy=config.SAVE_STRATEGY,
    save_total_limit=config.SAVE_TOTAL_LIMIT,
    eval_strategy=config.EVAL_STRATEGY,
    load_best_model_at_end=config.LOAD_BEST_MODEL,
    metric_for_best_model="loss",
    max_steps=getattr(config, "MAX_STEPS", -1),  # optional
)

# -------------------------
# Trainer
# -------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_splits["train"],
    eval_dataset=dataset_splits.get("validation", None),
    dataset_text_field="text",                   # created via chat template + semantic pass
    max_seq_length=config.MAX_SEQ_LENGTH,
    dataset_num_proc=DATASET_NUM_PROC,
    packing=config.PACKING,                      # if True, sequences may be packed up to max_seq_length
    args=training_args,
)

# -------------------------
# Train
# -------------------------
print("\nStarting training...")
trainer_stats = trainer.train()

# -------------------------
# Save full model + tokenizer
# -------------------------
print("\nSaving model...")
model.save_pretrained(config.MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(config.MODEL_OUTPUT_DIR)

print("\nTraining complete!")
print(f"Training stats: {trainer_stats}")

# -------------------------
# Save LoRA adapters only (PEFT weights)
# -------------------------
print("\nSaving LoRA adapters...")
model.save_pretrained(config.LORA_OUTPUT_DIR)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"\nModel saved to: {config.MODEL_OUTPUT_DIR}")
print(f"LoRA adapters saved to: {config.LORA_OUTPUT_DIR}")
print(f"Checkpoints saved to: {config.OUTPUT_DIR}")
