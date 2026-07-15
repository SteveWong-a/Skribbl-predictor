import requests
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vocab_path = os.path.join(base_dir, "data", "vocab.txt")

# 1. Read existing vocab
with open(vocab_path, "r") as f:
    existing_vocab = set(line.strip().lower() for line in f if line.strip())

# 2. Fetch official QuickDraw categories
url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
r = requests.get(url)
official_categories = [c.strip().lower() for c in r.text.split('\n') if c.strip()]

# 3. Add missing categories
new_words = 0
with open(vocab_path, "a") as f:
    for cat in official_categories:
        if cat not in existing_vocab:
            f.write(cat + "\n")
            new_words += 1

print(f"Added {new_words} new QuickDraw categories to vocab.txt!")
