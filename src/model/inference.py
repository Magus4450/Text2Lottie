#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.model.semantic_tokenizer import to_semantic, from_semantic
import src.model.config as config

# -----------------------------
# Load model and tokenizer
# -----------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_OUTPUT_DIR)
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_OUTPUT_DIR,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()

# -----------------------------
# Inference function
# -----------------------------
def generate_response(prompt: str, max_new_tokens=config.INFERENCE_MAX_NEW_TOKENS, temperature=0.7, top_p=0.9):
    """Generate a completion from the model given a text prompt."""
    # If your model was trained with chat templates, build a chat message
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Optional: semantic encoding if you used to_semantic in training
    input_text = to_semantic(input_text)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.INFERENCE_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    res = from_semantic(gen_text)

    return res

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    prompt = "What does the following lottie JSON animation represent?\n```json\n{\"fr\":60,\"ip\":0,\"op\":120,\"w\":512,\"h\":512,\"assets\":[],\"layers\":[{\"ind\":1,\"ty\":4,\"ks\":{\"o\":{\"a\":0,\"k\":100},\"r\":{\"a\":1,\"k\":[{\"t\":0,\"s\":[0],\"e\":[0],\"i\":{\"x\":[0.67],\"y\":[1.0]},\"o\":{\"x\":[0.33],\"y\":[0.0]}},{\"t\":120}]},\"p\":{\"a\":1,\"k\":[{\"t\":0,\"s\":[256.0,256.0],\"e\":[256.0,256.0],\"i\":{\"x\":[0.67,0.67],\"y\":[1.0,1.0]},\"o\":{\"x\":[0.33,0.33],\"y\":[0.0,0.0]}},{\"t\":120}]},\"a\":{\"a\":0,\"k\":[0,0,0]},\"s\":{\"a\":1,\"k\":[{\"t\":0,\"s\":[100,100,100],\"e\":[150.0,150.0,100],\"i\":{\"x\":[0.67,0.67,0.67],\"y\":[1.0,1.0,1.0]},\"o\":{\"x\":[0.33,0.33,0.33],\"y\":[0.0,0.0,0.0]}},{\"t\":120}]}},\"ao\":0,\"shapes\":[{\"ty\":\"gr\",\"it\":[{\"ty\":\"el\",\"p\":{\"a\":0,\"k\":[0,0]},\"s\":{\"a\":0,\"k\":[100,100]}},{\"ty\":\"st\",\"c\":{\"a\":0,\"k\":[0.0,0.0,0.0,1]},\"o\":{\"a\":0,\"k\":100},\"w\":{\"a\":0,\"k\":8},\"lc\":2,\"lj\":2,\"ml\":4},{\"ty\":\"tr\",\"p\":{\"a\":0,\"k\":[0,0]},\"a\":{\"a\":0,\"k\":[0,0]},\"s\":{\"a\":0,\"k\":[100,100]},\"r\":{\"a\":1,\"k\":[{\"t\":0,\"s\":[0],\"e\":[-90.0],\"i\":{\"x\":[0.67],\"y\":[1.0]},\"o\":{\"x\":[0.33],\"y\":[0.0]}},{\"t\":120}]},\"o\":{\"a\":0,\"k\":100}}]}],\"ip\":0,\"op\":120,\"st\":0}]}\n```"
    print(f"\nPrompt:\n{prompt}")
    output = generate_response(prompt)
    print("\nModel output:\n", output)
