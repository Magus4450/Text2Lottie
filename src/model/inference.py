#!/usr/bin/env python3
"""
Inference script for the fine-tuned Lottie model.

Usage
-----
python src/model/infer.py --prompt "Generate a lottie JSON animation of a bouncing ball."
python src/model/infer.py --chat examples/chat_example.json
"""

import os
import json
import torch
import argparse
from transformers import AutoTokenizer, pipeline
from unsloth import FastLanguageModel
from src.model.semantic_tokenizer import LottieSemanticTokenizer, to_semantic
import src.model.config as config


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Lottie model inference script")
    parser.add_argument("--prompt", type=str, help="User prompt for single-turn inference")
    parser.add_argument("--chat", type=str, help="Path to a chat JSON file (list of {role, content})")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--save", type=str, default=None, help="Save output JSON to a file if specified")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Load model & tokenizer
# -----------------------------------------------------------------------------
def load_model_and_tokenizer():
    print(f"Loading model from: {config.MODEL_OUTPUT_DIR}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_OUTPUT_DIR,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        load_in_4bit=config.LOAD_IN_4BIT,
    )

    # Right padding for causal models
    tokenizer.padding_side = "right"

    # Add semantic tokenizer augmentation
    print("Augmenting tokenizer with Lottie semantic tags...")
    tokenizer = LottieSemanticTokenizer(tokenizer, add_as_special_tokens=True).tokenizer
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


# -----------------------------------------------------------------------------
# Build chat template text
# -----------------------------------------------------------------------------
def build_chat_text(messages, tokenizer):
    """Apply chat template and semantic conversion."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = to_semantic(text)
    return text


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
@torch.inference_mode()
def generate(model, tokenizer, prompt_text, args):
    """Run model inference with generation parameters."""
    input_ids = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    output = model.generate(
        **input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer()

    if not args.prompt and not args.chat:
        raise ValueError("You must provide either --prompt or --chat")

    if args.chat:
        with open(args.chat, "r", encoding="utf-8") as f:
            messages = json.load(f)
        if not isinstance(messages, list):
            raise ValueError("Chat file must contain a list of {'role', 'content'} messages.")
        input_text = build_chat_text(messages, tokenizer)
    else:
        # Single prompt → chat format
        input_text = build_chat_text(
            [{"role": "user", "content": args.prompt.strip()}],
            tokenizer,
        )

    print("\n=== INPUT PROMPT ===\n")
    print(input_text)
    print("\n=== GENERATING... ===\n")

    output_text = generate(model, tokenizer, input_text, args)

    # Extract only the assistant part if the model outputs the entire chat
    if "assistant" in output_text.lower():
        split_idx = output_text.lower().find("assistant")
        output_text = output_text[split_idx + len("assistant"):].strip()

    print("=== MODEL OUTPUT ===\n")
    print(output_text)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"prompt": input_text, "output": output_text}, f, ensure_ascii=False, indent=2)
        print(f"\n[✓] Output saved to {args.save}")


if __name__ == "__main__":
    main()
