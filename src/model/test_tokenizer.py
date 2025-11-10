# test_semantic_tokenizer.py
import torch
from transformers import AutoTokenizer
from src.model.semantic_tokenizer import LottieSemanticTokenizer, to_semantic
import src.model.config as config

# 1. Load a base tokenizer (change this if needed)
BASE_MODEL = config.MODEL_NAME
print(f"Loading base tokenizer: {BASE_MODEL}")
base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)

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
```json
{"ty":"gr","a":0,"k":[{"ty":"sh"},{"ty":"fl"}]}
<|eot_id|>
"""

print("\nOriginal sample:\n", sample)

# Apply semantic key conversion

semantic_sample = to_semantic(sample)
print("\nAfter to_semantic():\n", semantic_sample)

# Encode & decode

encoded = base_tok(semantic_sample, add_special_tokens=False)
print("\nEncoded token IDs:", encoded["input_ids"][:50])
print("Decoded text:\n", base_tok.decode(encoded["input_ids"]))