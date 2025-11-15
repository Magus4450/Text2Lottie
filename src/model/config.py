
# Model Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct" 
NICKNAME = "llama_32_3B_MASKED"

MAX_SEQ_LENGTH = 3072  # Maximum sequence length (reduce if OOM)
LOAD_IN_4BIT = False    # Use 4-bit quantization (recommended)
DTYPE = "float16"           # None for auto, "float16" or "bfloat16"

# LoRA Configuration
LORA_R = 16            # LoRA rank (higher = more parameters, better quality)
LORA_ALPHA = 16        # LoRA alpha (typically same as rank)
LORA_DROPOUT = 0       # LoRA dropout (0 is optimized)
TARGET_MODULES = [     # Which layers to apply LoRA to
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens",
]

# Training Configuration
BATCH_SIZE = 2                    # Per device batch size
GRADIENT_ACCUMULATION_STEPS = 4   # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
NUM_EPOCHS = 2                    # Number of training epochs
LEARNING_RATE = 2e-4              # Learning rate
WARMUP_STEPS = 5                  # Number of warmup steps
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01               # Weight decay for regularization
OPTIMIZER = "adamw_bnb_8bit"          # Optimizer (adamw_8bit saves memory)
LR_SCHEDULER = "linear"           # Learning rate scheduler

# Data Configuration
DATASET_JSONL = "instruction_dataset.jsonl"
DATASET_TRAIN = "train.jsonl"
DATASET_TEST = "test_dup.jsonl"
DATASET_VAL = "val.jsonl"
DATASET_NUM_PROC = 24  # map workers
TRAIN_SPLIT = 0.9                 # 80% for training
VAL_SPLIT = 0.05                   # 10% for validation
TEST_SPLIT = 0.05                  # 10% for testing
SHUFFLE_SEED = 3407               # Random seed for shuffling



# Output Configuration
OUTPUT_DIR = f"outputs_{NICKNAME}"                      # Training checkpoints directory
MODEL_OUTPUT_DIR = f"lottie_model_{NICKNAME}"          # Final model directory
LORA_OUTPUT_DIR = f"lottie_model_lora_{NICKNAME}"      # LoRA adapters only directory
SAVE_STRATEGY = "steps"                     # When to save checkpoints
SAVE_TOTAL_LIMIT = 50                        # Maximum number of checkpoints to keep

# Evaluation Configuration
EVAL_STRATEGY = "steps"           # When to evaluate
EVAL_STEPS = 50
SAVE_STEPS = 100
METRIC_FOR_BEST_MODEL="eval_loss"
LOAD_BEST_MODEL = True           # Load best model at end based on eval loss

# Logging Configuration
LOGGING_STEPS = 10                 # How often to log
REPORT_TO = ["wandb"]               # "none", "wandb", "tensorboard"

# Advanced Settings
USE_GRADIENT_CHECKPOINTING = True # Save memory at cost of speed
PACKING = False                          # Pack multiple samples (faster for short sequences)

# Inference Configuration
INFERENCE_MAX_NEW_TOKENS = 4096   # Max tokens to generate during inference
INFERENCE_TEMPERATURE = 0.7       # Sampling temperature (lower = more deterministic)
INFERENCE_TOP_P = 0.9             # Top-p sampling
INFERENCE_DO_SAMPLE = True        # Whether to use sampling


def print_config():
    """Print current configuration."""
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
    print(f"4-bit Quantization: {LOAD_IN_4BIT}")
    print(f"\nLoRA Rank: {LORA_R}")
    print(f"LoRA Alpha: {LORA_ALPHA}")
    print(f"\nBatch Size: {BATCH_SIZE}")
    print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"\nData Split: {TRAIN_SPLIT*100:.0f}% train, {VAL_SPLIT*100:.0f}% val, {TEST_SPLIT*100:.0f}% test")
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print(f"Model Output: {MODEL_OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_config()

