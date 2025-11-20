import os
import re
import json
import torch
from ast import literal_eval
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from src.model.semantic_tokenizer import to_semantic, from_semantic, LottieSemanticTokenizer
import src.model.config as config

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("No CUDA device detected.")

MODEL_NAME = "outputs_llama_32_3b_TOKENIZED_BASE/checkpoint-900"
OUTPUT_DIR = f"{MODEL_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# -----------------------------
# # Augment tokenizer with Lottie semantic tags
# # -----------------------------
# print("Augmenting tokenizer with Lottie semantic tags/patterns...")
# _ = LottieSemanticTokenizer(tokenizer, add_as_special_tokens=False)

# # Ensure correct pad token
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '<pad>'})
# tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# -----------------------------
# Load base model + LoRA adapter
# -----------------------------
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
model = PeftModel.from_pretrained(base_model, MODEL_NAME)
model.eval()

# -----------------------------
# Generation function
# -----------------------------
def generate_response(prompt: str, max_new_tokens: int = 1024*5, temperature: float = 0.7):
    """
    Generate Lottie JSON or natural-language response from prompt.
    """
    # Convert to semantic if needed
    semantic_prompt = to_semantic(prompt)

    # Apply chat template if dataset used messages format
    chat = [{"role": "user", "content": semantic_prompt}]
    text_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(
        text_input,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
    ).to(model.device)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,
    )

    with torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=gen_cfg)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    res = from_semantic(generated_text)
    return res, text_input

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    with open("BASELINE_data/test.jsonl", "r") as test_set:
        lines = test_set.readlines()
    

    gold_path = os.path.join(OUTPUT_DIR, "gold")
    os.makedirs(gold_path, exist_ok=True)
    gen_path = os.path.join(OUTPUT_DIR, "gen")
    os.makedirs(gen_path, exist_ok=True)

    for i, line in enumerate(lines):
        print(f"[{i+1}/{len(lines)}]", end=" ")
        j_line = literal_eval(line)
        base_name = "-".join(j_line['id'].split("::")[-3:])
        if os.path.exists(os.path.join(gen_path, f"{base_name}.json")):
            print(f"Skipping {base_name}")
            continue
        
        prompt = j_line["messages"][0]["content"]
        gold_lottie = j_line['messages'][1]['content']

        # generate
        res, text_input = generate_response(prompt)
        pattern = r"<\|.*?\|>"
        text_input = re.sub(pattern, "", text_input, flags=re.DOTALL)
        output = res.replace(text_input, "")

        with open(os.path.join(gen_path, f"{base_name}.json"), "w") as gen_out:
            gen_out.write(output)
        
        with open(os.path.join(gold_path, f"{base_name}.json"), "w") as f:
            f.write(gold_lottie)

        print(f"Finished {base_name}")
