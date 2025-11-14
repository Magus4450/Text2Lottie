#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Dict

import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# Try PEFT-aware loader if available
try:
    from peft import AutoPeftModelForCausalLM  # type: ignore
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

# Project utilities / config
from src.model.semantic_tokenizer import LottieSemanticTokenizer, to_semantic
import src.model.config as config


def parse_args():
    p = argparse.ArgumentParser(description="Batch inference over a dataset split.")
    p.add_argument("--split", type=str, required=True,
                   choices=["train", "validation", "val", "test"],
                   help="Which split to run on.")
    p.add_argument("--model_name", type=str, required=True,
                   help="HF model or PEFT adapter path to load for inference.")
    p.add_argument("--save_folder", type=str, required=True,
                   help="Folder where {split}.json will be written.")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size for generation.")
    p.add_argument("--max_new_tokens", type=int, default=getattr(config, "GEN_MAX_NEW_TOKENS", 1024))
    p.add_argument("--temperature", type=float, default=getattr(config, "GEN_TEMPERATURE", 0.0))
    p.add_argument("--top_p", type=float, default=getattr(config, "GEN_TOP_P", 1.0))
    p.add_argument("--seed", type=int, default=getattr(config, "SEED", 42))
    p.add_argument("--limit", type=int, default=0, help="If >0, only run on the first N rows.")
    return p.parse_args()


def resolve_split_name(s: str) -> str:
    return "validation" if s.lower() in {"val", "validation"} else s.lower()


def load_splits() -> Dict[str, str]:
    """
    Mirror the training script's dataset path resolution from config.
    """
    DATASET_TRAIN = getattr(config, "DATASET_TRAIN", "train.jsonl")
    DATASET_VAL   = getattr(config, "DATASET_VAL", "val.jsonl")
    DATASET_TEST  = getattr(config, "DATASET_TEST", "test.jsonl")

    data_files = {}
    if os.path.exists(DATASET_TRAIN):
        data_files["train"] = DATASET_TRAIN
    if os.path.exists(DATASET_VAL):
        data_files["validation"] = DATASET_VAL
    if os.path.exists(DATASET_TEST):
        data_files["test"] = DATASET_TEST

    if not data_files:
        raise FileNotFoundError("No dataset files found. Expected train/val/test JSONL paths in config.")

    return data_files


def build_prompt(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Use the same chat template flow as training, then apply semantic mapping.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # we want the assistant to continue
    )
    text = to_semantic(text)
    return text


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    split = resolve_split_name(args.split)
    os.makedirs(args.save_folder, exist_ok=True)
    save_path = os.path.join(args.save_folder, f"{split}.json")

    # Load model/tokenizer (PEFT if available)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    print("Augmenting tokenizer with Lottie semantic tags/patterns...")
    _ = LottieSemanticTokenizer(tokenizer, add_as_special_tokens=False)

    # Ensure pad token
    if tokenizer.pad_token is None:
        print("Tokenizer has no pad_token; adding <pad> token.")
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )

    old_vocab = base_model.get_input_embeddings().weight.size(0)
    new_vocab = len(tokenizer)
    print(f"Resizing embeddings: {old_vocab} → {new_vocab}")
    if new_vocab != old_vocab:
        base_model.resize_token_embeddings(new_vocab, mean_resizing=False)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    print("Loading LoRA adapter weights...")
    # model = PeftModel.from_pretrained(base_model, config.MODEL_OUTPUT_DIR)
    model = PeftModel.from_pretrained(base_model, "lottie_model_llama_32_3B")
    model.eval()

    # Load datasets similarly to training
    data_files = load_splits()
    print(f"Loading dataset from files: {data_files}")
    dataset_splits = load_dataset("json", data_files=data_files)

    if split not in dataset_splits:
        raise ValueError(f"Split '{split}' not found in loaded dataset. Available: {list(dataset_splits.keys())}")

    ds = dataset_splits[split]

    # Filter malformed rows to mirror training
    ds = ds.filter(lambda ex: isinstance(ex.get("messages", None), list) and len(ex["messages"]) >= 2)

    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    print(f"Running inference on {len(ds)} examples from split='{split}'")

    results = []
    max_len = getattr(config, "MAX_SEQ_LENGTH", 2048)

    # Batched loop (manual to keep per-example prompt length tracking)
    def chunks(iterable, n):
        for i in range(0, len(iterable), n):
            yield range(i, min(i + n, len(iterable)))

    with torch.no_grad():
        for idx_range in chunks(range(len(ds)), args.batch_size):
            batch_prompts: List[str] = []
            ids: List[str] = []
            input_ids_list = []

            for i in idx_range:
                row = ds[i]
                ids.append(row.get("id", str(i)))
                prompt = build_prompt(tokenizer, row["messages"])
                batch_prompts.append(prompt)

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)

            # Track the prompt lengths so we can slice generated continuation
            prompt_lens = attention_mask.sum(dim=1).tolist()

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0.0,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            # Decode only the continuation beyond each prompt
            for j, out_ids in enumerate(outputs):
                cont_ids = out_ids[prompt_lens[j]:]
                text = tokenizer.decode(cont_ids, skip_special_tokens=True)
                # Optional: strip leading spaces/newlines
                text = text.lstrip()
                results.append({
                    "id": ids[j],
                    "prediction": text,
                })

            if len(results) % 100 == 0:
                print(f"Generated {len(results)}/{len(ds)}")

    # Save results
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} predictions to: {save_path}")


if __name__ == "__main__":
    main()
