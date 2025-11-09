"""
Configuration file for fine-tuning.
Modify these settings according to your needs.
"""

# Model Configuration
MODEL_NAME = "unsloth/Qwen3-8B"  # Change if not available
# Alternatives:
#   - "unsloth/Qwen2.5-Coder-7B-Instruct" (recommended for JSON)
#   - "unsloth/Qwen2.5-7B-Instruct"
#   - "unsloth/llama-3-8b-Instruct"
#   - "unsloth/mistral-7b-instruct-v0.3"

MAX_SEQ_LENGTH = 8192  # Maximum sequence length (reduce if OOM)
LOAD_IN_4BIT = True    # Use 4-bit quantization (recommended)
DTYPE = None           # None for auto, "float16" or "bfloat16"

# LoRA Configuration
LORA_R = 16            # LoRA rank (higher = more parameters, better quality)
LORA_ALPHA = 16        # LoRA alpha (typically same as rank)
LORA_DROPOUT = 0       # LoRA dropout (0 is optimized)
TARGET_MODULES = [     # Which layers to apply LoRA to
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training Configuration
BATCH_SIZE = 16                    # Per device batch size
GRADIENT_ACCUMULATION_STEPS = 4   # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
NUM_EPOCHS = 3                    # Number of training epochs
LEARNING_RATE = 2e-4              # Learning rate
WARMUP_STEPS = 5                  # Number of warmup steps
WEIGHT_DECAY = 0.01               # Weight decay for regularization
OPTIMIZER = "adamw_8bit"          # Optimizer (adamw_8bit saves memory)
LR_SCHEDULER = "linear"           # Learning rate scheduler

# Data Configuration
DATASET_JSONL = "instruction_dataset.jsonl"
DATASET_NUM_PROC = 4  # map workers
TRAIN_SPLIT = 0.8                 # 80% for training
VAL_SPLIT = 0.1                   # 10% for validation
TEST_SPLIT = 0.1                  # 10% for testing
SHUFFLE_SEED = 3407               # Random seed for shuffling



# Output Configuration
OUTPUT_DIR = "outputs"                      # Training checkpoints directory
MODEL_OUTPUT_DIR = "lottie_model"          # Final model directory
LORA_OUTPUT_DIR = "lottie_model_lora"      # LoRA adapters only directory
SAVE_STRATEGY = "epoch"                     # When to save checkpoints
SAVE_TOTAL_LIMIT = 2                        # Maximum number of checkpoints to keep

# Evaluation Configuration
EVAL_STRATEGY = "epoch"           # When to evaluate
LOAD_BEST_MODEL = True           # Load best model at end based on eval loss

# Logging Configuration
LOGGING_STEPS = 1                 # How often to log
REPORT_TO = "none"               # "none", "wandb", "tensorboard"

# Advanced Settings
USE_GRADIENT_CHECKPOINTING = "unsloth"  # Save memory at cost of speed
PACKING = False                          # Pack multiple samples (faster for short sequences)
DATASET_NUM_PROC = 2                     # Number of processes for data loading

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

