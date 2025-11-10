"""
Evaluation script to test the fine-tuned model on the test set.
"""

import torch
from unsloth import FastLanguageModel
import json
from tqdm import tqdm
import config
from dataset import get_test_data

print("="*60)
print("Model Evaluation Script")
print("="*60)

# Load model
print("\nLoading fine-tuned model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.MODEL_OUTPUT_DIR,
    max_seq_length=config.MAX_SEQ_LENGTH,
    dtype=config.DTYPE,
    load_in_4bit=config.LOAD_IN_4BIT,
)

# Enable inference mode
FastLanguageModel.for_inference(model)
print("Model loaded successfully!")

# Load test data
print("\nLoading test data...")
test_data = get_test_data(
    data_dir=config.DATA_DIR,
    shuffle_seed=config.SHUFFLE_SEED,
    train_split=config.TRAIN_SPLIT,
    val_split=config.VAL_SPLIT,
    test_split=config.TEST_SPLIT
)
print(f"Test set size: {len(test_data)} samples")

# Evaluate on a sample of test set
print("\nEvaluating model...")
num_samples = min(10, len(test_data))  # Test on 10 samples
print(f"Testing on {num_samples} samples from test set...")

results = []

for i in tqdm(range(num_samples), desc="Generating"):
    sample = test_data[i]
    prompt = sample["caption"]
    target = sample["target"]
    
    # Format the prompt
    messages = [
        {"role": "user", "content": f"Generate a Lottie Json which does the following: {prompt}"}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.INFERENCE_MAX_NEW_TOKENS,
            use_cache=True,
            temperature=config.INFERENCE_TEMPERATURE,
            top_p=config.INFERENCE_TOP_P,
            do_sample=config.INFERENCE_DO_SAMPLE,
        )
    
    generated_text = tokenizer.batch_decode(outputs)[0]
    
    # Extract assistant response
    try:
        assistant_response = generated_text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
    except:
        assistant_response = generated_text
    
    # Simple metrics
    target_len = len(target)
    generated_len = len(assistant_response)
    
    # Check if it's valid JSON
    is_valid_json = False
    try:
        json.loads(assistant_response)
        is_valid_json = True
    except:
        pass
    
    results.append({
        "sample": i,
        "prompt": prompt,
        "target_length": target_len,
        "generated_length": generated_len,
        "is_valid_json": is_valid_json,
        "generated": assistant_response,
        "target": target
    })

# Print results
print("\n" + "="*60)
print("Evaluation Results")
print("="*60)

valid_json_count = sum(1 for r in results if r["is_valid_json"])
print(f"\nValid JSON generated: {valid_json_count}/{num_samples} ({valid_json_count/num_samples*100:.1f}%)")

avg_target_len = sum(r["target_length"] for r in results) / len(results)
avg_generated_len = sum(r["generated_length"] for r in results) / len(results)
print(f"\nAverage target length: {avg_target_len:.0f} characters")
print(f"Average generated length: {avg_generated_len:.0f} characters")
print(f"Length ratio: {avg_generated_len/avg_target_len:.2f}x")

# Show sample results
print("\n" + "="*60)
print("Sample Results")
print("="*60)

for i, result in enumerate(results[:3], 1):
    print(f"\n--- Sample {i} ---")
    print(f"Prompt: {result['prompt'][:100]}...")
    print(f"Valid JSON: {result['is_valid_json']}")
    print(f"Generated length: {result['generated_length']} chars")
    print(f"First 200 chars of generation:")
    print(result['generated'][:200] + "...")

# Save detailed results
results_file = "evaluation_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\nDetailed results saved to: {results_file}")
print("\n" + "="*60)
print("Evaluation Complete!")
print("="*60 + "\n")

