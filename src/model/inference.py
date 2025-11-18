import torch
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

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)

# -----------------------------
# Augment tokenizer with Lottie semantic tags
# -----------------------------
print("Augmenting tokenizer with Lottie semantic tags/patterns...")
_ = LottieSemanticTokenizer(tokenizer, add_as_special_tokens=False)

# Ensure correct pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

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
model = PeftModel.from_pretrained(base_model, "outputs_llama_32_3B_MASKED_NO_LEAK/checkpoint-1800")
model.eval()

# -----------------------------
# Generation function
# -----------------------------
def generate_response(prompt: str, max_new_tokens: int = 4096, temperature: float = 0.7):
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
    return res

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # prompt = "What does the following lottie JSON animation represent?\n```json\n{\"fr\":60,\"ip\":0,\"op\":120,\"w\":512,\"h\":512,\"assets\":[],\"layers\":[{\"ind\":1,\"ty\":4,\"ks\":{\"o\":{\"a\":0,\"k\":100},\"r\":{\"a\":1,\"k\":[{\"t\":0,\"s\":[0],\"e\":[0],\"i\":{\"x\":[0.67],\"y\":[1.0]},\"o\":{\"x\":[0.33],\"y\":[0.0]}},{\"t\":120}]},\"p\":{\"a\":1,\"k\":[{\"t\":0,\"s\":[256.0,256.0],\"e\":[256.0,256.0],\"i\":{\"x\":[0.67,0.67],\"y\":[1.0,1.0]},\"o\":{\"x\":[0.33,0.33],\"y\":[0.0,0.0]}},{\"t\":120}]},\"a\":{\"a\":0,\"k\":[0,0,0]},\"s\":{\"a\":1,\"k\":[{\"t\":0,\"s\":[100,100,100],\"e\":[150.0,150.0,100],\"i\":{\"x\":[0.67,0.67,0.67],\"y\":[1.0,1.0,1.0]},\"o\":{\"x\":[0.33,0.33,0.33],\"y\":[0.0,0.0,0.0]}},{\"t\":120}]}},\"ao\":0,\"shapes\":[{\"ty\":\"gr\",\"it\":[{\"ty\":\"el\",\"p\":{\"a\":0,\"k\":[0,0]},\"s\":{\"a\":0,\"k\":[100,100]}},{\"ty\":\"st\",\"c\":{\"a\":0,\"k\":[0.0,0.0,0.0,1]},\"o\":{\"a\":0,\"k\":100},\"w\":{\"a\":0,\"k\":8},\"lc\":2,\"lj\":2,\"ml\":4},{\"ty\":\"tr\",\"p\":{\"a\":0,\"k\":[0,0]},\"a\":{\"a\":0,\"k\":[0,0]},\"s\":{\"a\":0,\"k\":[100,100]},\"r\":{\"a\":1,\"k\":[{\"t\":0,\"s\":[0],\"e\":[-90.0],\"i\":{\"x\":[0.67],\"y\":[1.0]},\"o\":{\"x\":[0.33],\"y\":[0.0]}},{\"t\":120}]},\"o\":{\"a\":0,\"k\":100}}]}],\"ip\":0,\"op\":120,\"st\":0}]}\n```"
    prompts = {
        "normal": "Generate a lottie JSON animation given the following description: ",
        "static": "Given a static lottie JSON animation, add animation with the given description.\n\nStatic:\n",
        "rev": "What does the following lottie JSON animation represent?\n"
    }

    while True:
        type_ = input("type>")
        user = input("prompt>")
        prompt = prompts[type_] + user

        print(f"\nPrompt:\n{prompt}")
        output = generate_response(prompt)
        print("\nModel output:\n", output)
