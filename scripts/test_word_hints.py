import os
import random
import torch
import torch.nn.functional as F

def load_vocab(vocab_path):
    if not os.path.exists(vocab_path):
        print(f"Error: vocab file not found at {vocab_path}")
        return []
    with open(vocab_path, 'r') as f:
        vocab = [line.strip().lower() for line in f if line.strip()]
    return vocab

def filter_words_by_hint(vocab, hint):
    """
    Filters the vocabulary based on a hint pattern.
    The hint uses underscores '_' for unknown characters.
    Example: "__pl_" for a 5-letter word matching "apple"
    """
    valid_words = []
    for word in vocab:
        if len(word) != len(hint):
            continue
        
        match = True
        for i in range(len(word)):
            if hint[i] != '_' and hint[i] != word[i]:
                match = False
                break
                
        if match:
            valid_words.append(word)
            
    return valid_words

def simulate_prediction(vocab, hint, target_word=None):
    print(f"\n{'='*50}")
    print(f"Simulating Prediction for hint: '{hint}'")
    print(f"{'='*50}")
    
    # 1. Filter words
    valid_words = filter_words_by_hint(vocab, hint)
    print(f"Found {len(valid_words)} valid words matching the hint pattern.")
    
    if not valid_words:
        print("No valid words found!")
        return
        
    if len(valid_words) <= 10:
        print(f"Valid words: {valid_words}")
    else:
        print(f"Valid words: {valid_words[:5]} ... (and {len(valid_words)-5} more)")
        
    # 2. Mock Embeddings
    # We will give the target word a slightly higher similarity, and the rest random.
    # In reality, this comes from the model.
    print("\nCalculating simulated probabilities...")
    logits = []
    
    # Make up a random embedding for the "image"
    output_emb = torch.randn(300)
    
    word_embeddings = {}
    for word in valid_words:
        # If this is the correct word, make its embedding closer to the output
        if word == target_word:
            word_emb = output_emb + torch.randn(300) * 0.5 
        else:
            word_emb = torch.randn(300)
        word_embeddings[word] = word_emb
        
        sim = F.cosine_similarity(output_emb, word_emb, dim=0)
        logits.append(sim)
        
    logits_tensor = torch.stack(logits)
    
    # Apply scaled softmax (temperature scaling, as in predictor.py)
    probabilities = torch.softmax(logits_tensor * 10.0, dim=0).tolist()
    
    similarities = [(valid_words[i], probabilities[i]) for i in range(len(valid_words))]
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Predictions:")
    for i, (word, prob) in enumerate(similarities[:5]):
        print(f"{i+1}. {word} ({prob*100:.2f}%)")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_path = os.path.join(base_dir, "data", "vocab.txt")
    
    vocab = load_vocab(vocab_path)
    if not vocab:
        return
        
    print(f"Loaded {len(vocab)} total vocabulary words.")
    
    # Test Case 1: Only word length known (5 letters)
    simulate_prediction(vocab, "_____", target_word="apple")
    
    # Test Case 2: One letter revealed
    simulate_prediction(vocab, "a____", target_word="apple")
    
    # Test Case 3: More letters revealed
    simulate_prediction(vocab, "a__le", target_word="apple")

if __name__ == "__main__":
    main()
