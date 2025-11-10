# test_semantic_tokenizer.py
import torch
from transformers import AutoTokenizer
from src.model.semantic_tokenizer import LottieSemanticTokenizer, to_semantic, from_semantic
import src.model.config as config
import copy

# 1. Load a base tokenizer (change this if needed)
BASE_MODEL = config.MODEL_NAME
print(f"Loading base tokenizer: {BASE_MODEL}")
base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)

actual_base_tok = copy.deepcopy(base_tok)

# 2. Wrap it with your semantic tokenizer
print("\nAugmenting tokenizer with Lottie semantic tags/patterns...")
lottie_tok = LottieSemanticTokenizer(base_tok, add_as_special_tokens=False)

# 3. Inspect which tokens were added
print(f"\nBase vocab size: {base_tok.vocab_size}")
print(f"New vocab size:  {len(base_tok)}")

# Manually compute the difference
added = [t for t in base_tok.get_vocab().keys() if t.startswith("<") and t.endswith(">")]
print(f"\nNumber of '<...>' tokens in vocab: {len(added)}")
print("Sample added tokens:", added[:20])

# 4. Test a semantic snippet
sample = """
<|begin_of_text|>
User: Generate a static Lottie JSON
Assistant:
{"id": "generated_data::normal::rev::square__up-to-down__ease-out__size-220px__fill__scale-0.5", "messages": [{"role": "user", "content": "What does the following lottie JSON animation represent?\n\n```json\n{\"fr\":60,\"ip\":0,\"op\":120,\"w\":512,\"h\":512,\"assets\":[],\"layers\":[{\"ind\":1,\"ty\":4,\"ks\":{\"o\":{\"a\":0,\"k\":100},\"r\":{\"a\":1,\"k\":[{\"t\":0,\"s\":[0],\"e\":[0],\"i\":{\"x\":[0.33],\"y\":[1.0]},\"o\":{\"x\":[0.67],\"y\":[0.0]}},{\"t\":120}]},\"p\":{\"a\":1,\"k\":[{\"t\":0,\"s\":[256.0,102.4],\"e\":[256.0,409.6],\"i\":{\"x\":[0.33,0.33],\"y\":[1.0,1.0]},\"o\":{\"x\":[0.67,0.67],\"y\":[0.0,0.0]}},{\"t\":120}]},\"a\":{\"a\":0,\"k\":[0,0,0]},\"s\":{\"a\":1,\"k\":[{\"t\":0,\"s\":[100,100,100],\"e\":[50.0,50.0,100],\"i\":{\"x\":[0.33,0.33,0.33],\"y\":[1.0,1.0,1.0]},\"o\":{\"x\":[0.67,0.67,0.67],\"y\":[0.0,0.0,0.0]}},{\"t\":120}]}},\"ao\":0,\"shapes\":[{\"ty\":\"gr\",\"it\":[{\"ty\":\"rc\",\"p\":{\"a\":0,\"k\":[0,0]},\"s\":{\"a\":0,\"k\":[220,220]},\"r\":{\"a\":0,\"k\":0}},{\"ty\":\"fl\",\"c\":{\"a\":0,\"k\":[0.0,0.0,0.0,1]},\"o\":{\"a\":0,\"k\":100}},{\"ty\":\"tr\",\"p\":{\"a\":0,\"k\":[0,0]},\"a\":{\"a\":0,\"k\":[0,0]},\"s\":{\"a\":0,\"k\":[100,100]},\"r\":{\"a\":0,\"k\":0},\"o\":{\"a\":0,\"k\":100}}]}],\"ip\":0,\"op\":120,\"st\":0}]}\n```"}, {"role": "assistant", "content": "A big filled square that moves from up to down and scales to 0.5× its size, with ease-out easing, over 2 seconds."}], "metadata": {"dataset": "generated_data", "category": "normal", "direction": "reverse", "key": "square__up-to-down__ease-out__size-220px__fill__scale-0.5"}}
<|eot_id|>
"""

print("\nOriginal sample:\n", sample)

# Apply semantic key conversion

semantic_sample = to_semantic(sample)
print("\nAfter to_semantic():\n", semantic_sample)

# Encode & decode

encoded = base_tok(semantic_sample, add_special_tokens=False)
base_encoded = actual_base_tok(semantic_sample, add_special_tokens=False)

print(len(encoded["input_ids"]))
print(len(base_encoded["input_ids"]))

print("\nEncoded token IDs:", encoded["input_ids"][:50])
print("Decoded text:\n", base_tok.decode(encoded["input_ids"]))
for tok in encoded["input_ids"]:
    print(tok, base_tok.decode(tok))
from_sample = from_semantic(base_tok.decode(encoded["input_ids"]))
print("Restored: ", from_sample)