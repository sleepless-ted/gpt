import os
from setup import GPTConfig, DATA_DIR
from collections import Counter
from tokenize_tools import tokenize, pattern

IN_FILE = os.path.join(DATA_DIR, 'input.txt')
CLEANED_FILE = os.path.join(DATA_DIR, 'input_clean.txt')

with open(IN_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

tokens = tokenize(text, pattern)
print(f"Found {len(tokens)} tokens, computing frequencies...")

freq = Counter(tokens)
most_common = [w for w, _ in freq.most_common(GPTConfig.vocab_size)]
vocab = set(most_common)
print(f"Keeping top {len(vocab)} tokens (most frequent).")

# rebuild the cleaned corpus
cleaned_tokens = [token for token in tokens if token in vocab]
cleaned = ' '.join(cleaned_tokens)

with open(CLEANED_FILE, 'w', encoding='utf-8') as f:
    f.write(cleaned)

print(f"Wrote {CLEANED_FILE} (size: {len(cleaned_tokens)} tokens).")