# pip install -U torch transformers peft accelerate
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER_ID = "xingxm/LLM4SVG-DeepSeekDistill-Qwen-Instruct-2401028"
BASE_ID    = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

def load_model_and_tokenizer(base_id=BASE_ID, adapter_id=ADAPTER_ID):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=dtype,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base,
        adapter_id,
        torch_dtype=dtype,
    )
    model.eval()
    return model, tokenizer

def generate_svg(prompt, model, tokenizer,
                 max_new_tokens=1024, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text

def extract_and_save_svg(text, path="out.svg"):
    # Try to pull a single <svg ...>...</svg> block if the model adds prose
    m = re.search(r"<svg[\\s\\S]*?</svg>", text, re.IGNORECASE)
    svg = m.group(0) if m else text
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return path, svg[:2000]  # return a preview

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()

    # Keep the instruction super explicit so the model outputs pure SVG.
    prompt = (
        "You are an SVG code generator. Output ONLY valid SVG markup.\n\n"
        "Task: Draw a simple birthday cake with three candles.\n"
        "Constraints: width=256, height=256, no external references, minimal fills and strokes."
    )

    text = generate_svg(prompt, model, tokenizer)
    path, preview = extract_and_save_svg(text, "out.svg")

    print(f"Saved SVG to: {path}")
    print("Preview:")
    print(preview)
