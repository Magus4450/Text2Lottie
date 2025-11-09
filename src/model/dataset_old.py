"""
Dataset creation and management for Lottie JSON fine-tuning.
This module handles loading, processing, and splitting the dataset.
"""

from datasets import Dataset, DatasetDict
from pathlib import Path
import json
import pandas as pd
from typing import Tuple, Optional, List, Dict
import config


def load_raw_data(data_dir: str = None) -> List[Dict[str, str]]:
    """
    Load raw data from lottie_data directory.
    
    Args:
        data_dir: Path to data directory. If None, uses config.DATA_DIR
        
    Returns:
        List of dictionaries with 'input' and 'target' keys
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    root = Path(data_dir)
    json_dir = root / "json"
    caption_dir = root / "caption"
    
    if not root.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found. "
            "Please run setup_data.sh or manually copy lottie_data to this directory."
        )
    
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    
    if not caption_dir.exists():
        raise FileNotFoundError(f"Caption directory not found: {caption_dir}")
    
    data = []
    
    # Loop through all JSON files
    for json_file in json_dir.glob("*.json"):
        base_name = json_file.stem
        caption_file = caption_dir / f"{base_name}.txt"
        
        if caption_file.exists():
            # Read caption text
            caption = caption_file.read_text().strip()
            # Read lottie JSON content
            lottie_json = json_file.read_text().strip()
            
            # Add to dataset
            data.append({
                "input": "Generate a Lottie Json which does the following: " + caption,
                "target": lottie_json,
                "caption": caption,  # Keep original caption for reference
                "file_id": base_name
            })
    
    if len(data) == 0:
        raise ValueError(
            f"No data found. Please ensure {data_dir}/json and "
            f"{data_dir}/caption directories contain matching files."
        )
    
    return data


def create_conversation_dataset(
    data: List[Dict[str, str]],
    tokenizer,
    shuffle_seed: int = None
) -> Dataset:
    """
    Convert raw data to conversation format and create dataset.
    
    Args:
        data: List of dictionaries with 'input' and 'target' keys
        tokenizer: Tokenizer to apply chat template
        shuffle_seed: Random seed for shuffling. If None, uses config.SHUFFLE_SEED
        
    Returns:
        Dataset with 'text' field containing formatted conversations
    """
    if shuffle_seed is None:
        shuffle_seed = config.SHUFFLE_SEED
    
    # Create a Hugging Face Dataset
    dataset = Dataset.from_list(data)
    
    def generate_conversation(examples):
        problems = examples["input"]
        solutions = examples["target"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            conversations.append([
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ])
        return {"conversations": conversations}
    
    # Generate conversations
    dataset_with_conversations = dataset.map(generate_conversation, batched=True)
    
    # Apply chat template
    conversations = tokenizer.apply_chat_template(
        dataset_with_conversations["conversations"],
        tokenize=False,
    )
    
    # Create final dataset with text field
    dataset = Dataset.from_pandas(pd.DataFrame(conversations, columns=["text"]))
    dataset = dataset.shuffle(seed=shuffle_seed)
    
    return dataset


def split_dataset(
    dataset: Dataset,
    train_split: float = None,
    val_split: float = None,
    test_split: float = None,
    seed: int = 42
) -> DatasetDict:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_split: Proportion for training. If None, uses config.TRAIN_SPLIT
        val_split: Proportion for validation. If None, uses config.VAL_SPLIT
        test_split: Proportion for test. If None, uses config.TEST_SPLIT
        seed: Random seed for splitting
        
    Returns:
        DatasetDict with 'train', 'validation', and 'test' keys
    """
    if train_split is None:
        train_split = config.TRAIN_SPLIT
    if val_split is None:
        val_split = config.VAL_SPLIT
    if test_split is None:
        test_split = config.TEST_SPLIT
    
    # Validate splits
    total = train_split + val_split + test_split
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"Splits must sum to 1.0, got {total} "
            f"(train={train_split}, val={val_split}, test={test_split})"
        )
    
    # First split: train vs (validation + test)
    train_testvalid = dataset.train_test_split(
        test_size=(val_split + test_split),
        seed=seed
    )
    
    # Second split: validation vs test
    if val_split + test_split > 0:
        test_valid_proportion = test_split / (val_split + test_split)
        test_valid = train_testvalid["test"].train_test_split(
            test_size=test_valid_proportion,
            seed=seed
        )
        
        dataset_splits = DatasetDict({
            "train": train_testvalid["train"],
            "validation": test_valid["train"],
            "test": test_valid["test"]
        })
    else:
        # No validation/test split needed
        dataset_splits = DatasetDict({
            "train": train_testvalid["train"],
            "validation": Dataset.from_list([]),
            "test": Dataset.from_list([])
        })
    
    return dataset_splits


def prepare_dataset(
    tokenizer,
    data_dir: str = None,
    shuffle_seed: int = None,
    train_split: float = None,
    val_split: float = None,
    test_split: float = None,
    verbose: bool = True
) -> DatasetDict:
    """
    Complete pipeline: load data, create conversations, and split.
    
    Args:
        tokenizer: Tokenizer to apply chat template
        data_dir: Path to data directory. If None, uses config.DATA_DIR
        shuffle_seed: Random seed for shuffling. If None, uses config.SHUFFLE_SEED
        train_split: Proportion for training. If None, uses config.TRAIN_SPLIT
        val_split: Proportion for validation. If None, uses config.VAL_SPLIT
        test_split: Proportion for test. If None, uses config.TEST_SPLIT
        verbose: Whether to print progress information
        
    Returns:
        DatasetDict with 'train', 'validation', and 'test' keys
    """
    if verbose:
        print("Loading raw data...")
    
    data = load_raw_data(data_dir)
    
    if verbose:
        print(f"Total samples loaded: {len(data)}")
        print(f"\nFirst sample:")
        print(f"  Caption: {data[0]['caption'][:100]}...")
        print(f"  Target length: {len(data[0]['target'])} characters")
    
    if verbose:
        print("\nCreating conversation dataset...")
    
    dataset = create_conversation_dataset(data, tokenizer, shuffle_seed)
    
    if verbose:
        print("Splitting dataset...")
    
    dataset_splits = split_dataset(
        dataset,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split
    )
    
    if verbose:
        print("\nDataset splits:")
        print(dataset_splits)
        for k, v in dataset_splits.items():
            print(f"  {k}: {len(v)} samples")
    
    return dataset_splits


def get_raw_data_splits(
    data_dir: str = None,
    shuffle_seed: int = None,
    train_split: float = None,
    val_split: float = None,
    test_split: float = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Get raw data (before conversation formatting) split into train/val/test.
    Useful for evaluation and inference where you need the original prompts.
    
    Args:
        data_dir: Path to data directory. If None, uses config.DATA_DIR
        shuffle_seed: Random seed for shuffling. If None, uses config.SHUFFLE_SEED
        train_split: Proportion for training. If None, uses config.TRAIN_SPLIT
        val_split: Proportion for validation. If None, uses config.VAL_SPLIT
        test_split: Proportion for test. If None, uses config.TEST_SPLIT
        
    Returns:
        Tuple of (train_data, val_data, test_data) as lists of dictionaries
    """
    if shuffle_seed is None:
        shuffle_seed = config.SHUFFLE_SEED
    if train_split is None:
        train_split = config.TRAIN_SPLIT
    if val_split is None:
        val_split = config.VAL_SPLIT
    if test_split is None:
        test_split = config.TEST_SPLIT
    
    # Load raw data
    data = load_raw_data(data_dir)
    
    # Create dataset and shuffle
    dataset = Dataset.from_list(data).shuffle(seed=shuffle_seed)
    
    # Split
    train_testvalid = dataset.train_test_split(
        test_size=(val_split + test_split),
        seed=42
    )
    
    if val_split + test_split > 0:
        test_valid_proportion = test_split / (val_split + test_split)
        test_valid = train_testvalid["test"].train_test_split(
            test_size=test_valid_proportion,
            seed=42
        )
        
        train_data = list(train_testvalid["train"])
        val_data = list(test_valid["train"])
        test_data = list(test_valid["test"])
    else:
        train_data = list(train_testvalid["train"])
        val_data = []
        test_data = []
    
    return train_data, val_data, test_data


# Convenience function for getting just test data
def get_test_data(
    data_dir: str = None,
    shuffle_seed: int = None,
    train_split: float = None,
    val_split: float = None,
    test_split: float = None
) -> List[Dict]:
    """
    Get only the test split of raw data.
    
    Returns:
        List of dictionaries with test data
    """
    _, _, test_data = get_raw_data_splits(
        data_dir=data_dir,
        shuffle_seed=shuffle_seed,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split
    )
    return test_data


if __name__ == "__main__":
    """Test the dataset creation pipeline."""
    print("Testing dataset creation pipeline...\n")
    
    # Test loading raw data
    print("1. Loading raw data...")
    try:
        data = load_raw_data()
        print(f"   ✓ Loaded {len(data)} samples")
        print(f"   Sample caption: {data[0]['caption'][:80]}...")
        print(f"   Sample target length: {len(data[0]['target'])} chars")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        exit(1)
    
    # Test splitting
    print("\n2. Testing data splits...")
    try:
        train_data, val_data, test_data = get_raw_data_splits()
        print(f"   ✓ Train: {len(train_data)} samples")
        print(f"   ✓ Validation: {len(val_data)} samples")
        print(f"   ✓ Test: {len(test_data)} samples")
        
        total = len(train_data) + len(val_data) + len(test_data)
        print(f"   ✓ Total: {total} samples")
        
        if total != len(data):
            print(f"   ✗ Warning: Split total ({total}) != original ({len(data)})")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        exit(1)
    
    print("\n✓ All tests passed!")
    print("\nTo use in training:")
    print("  from dataset import prepare_dataset")
    print("  dataset_splits = prepare_dataset(tokenizer)")

