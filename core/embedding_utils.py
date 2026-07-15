# embedding_utils.py
import numpy as np
import torch
import os

def load_glove_embeddings(vocab, glove_path="glove.6B.300d.txt"):
    print(f"Loading GloVe embeddings from {glove_path}...")
    
    # Initialize all with random vectors (fallback for words not found in GloVe)
    torch.manual_seed(42)
    embeddings = {word: torch.randn(300) for word in vocab}
    
    if not os.path.exists(glove_path):
        print(f"WARNING: {glove_path} not found. Using random embeddings. Please download GloVe.")
        return embeddings

    found_count = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = torch.from_numpy(vector)
                found_count += 1
                
    print(f"Successfully loaded {found_count}/{len(vocab)} words from GloVe.")
    return embeddings