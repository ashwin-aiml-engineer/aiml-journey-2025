"""Day 25.05 — Tokenization demo (transformers tokenizers safe stub)
Run time: ~10-15 minutes

- Shows tokenization steps: tokenize, convert to ids, padding/truncation
- Uses `transformers` if available, otherwise demonstrates basic whitespace tokenizer
"""

try:
    from transformers import AutoTokenizer
    has_tokenizers = True
except Exception:
    has_tokenizers = False

if has_tokenizers:
    def run_demo():
        t = AutoTokenizer.from_pretrained('bert-base-uncased')
        txt = "Hello world! This is a test."
        tokens = t(txt)
        print('Tokens:', tokens)
        print('Token IDs:', tokens['input_ids'][:10])
else:
    def run_demo():
        print('transformers not installed. Falling back to whitespace tokenizer.')
        text = "Hello world! This is a test."
        toks = text.split()
        ids = [hash(w) % 10000 for w in toks]
        print('Tokens:', toks)
        print('Pseudo-IDs:', ids)

if __name__ == '__main__':
    run_demo()

    # Exercises:
    # - Try padding and truncation examples with a tokenizer if available.
    # - Inspect tokenizer.special_tokens_map for special tokens.