import threading
import queue
import time
import numpy as np
import random
from core.embedding_utils import load_glove_embeddings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image

    def filter_words_by_hint(vocab, hint):
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
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch not available or corrupted: {e}\nRunning dummy predictor in mock mode.")

if TORCH_AVAILABLE:
    import torchvision.models as models

    class SkribblPredictorModel(nn.Module):
        def __init__(self, embedding_dim=300):
            super().__init__()
            self.resnet = models.resnet18(weights='DEFAULT')
            # Remove the final FC layer
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Identity()
            
            # We concatenate the word length (1 dim) to the 512 features
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(num_ftrs + 1, embedding_dim)
            
        def forward(self, x, lengths):
            features = self.resnet(x)
            
            # Convert lengths to tensor of shape (batch_size, 1) and concatenate
            lengths = lengths.unsqueeze(1).float()
            combined = torch.cat((features, lengths), dim=1)
            
            # Apply dropout to prevent overfitting
            combined = self.dropout(combined)
            
            return self.fc(combined)

class PredictorThread(threading.Thread):
    def __init__(self, vocab_path, input_queue, output_queue):
        super().__init__()
        self.daemon = True
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        
        self.load_vocab(vocab_path)
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
            self.model = SkribblPredictorModel(embedding_dim=300).to(self.device)
            
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "weights", "skribbl_model.pth")
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded trained model weights from {model_path}")
                
            self.model.eval()
            
            # Load the real embeddings and move them to the correct device (GPU/CPU)
            glove_path = os.path.join(base_dir, "data", "glove.6B.300d.txt")
            cpu_embeddings = load_glove_embeddings(self.vocab, glove_path)
            self.word_embeddings = {word: vec.to(self.device) for word, vec in cpu_embeddings.items()}
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r') as f:
            self.vocab = [line.strip() for line in f if line.strip()]
            
    def run(self):
        while self.running:
            try:
                item = self.input_queue.get(timeout=1.0)
                if item is None:
                    continue
                    
                image_array, target_hint = item
                
                if isinstance(target_hint, int):
                    target_hint = "_" * target_hint
                    
                target_length = len(target_hint)
                
                predictions = []
                if np.all(image_array == 255):
                    pass # Keep predictions empty
                elif TORCH_AVAILABLE:
                    tensor_img = self.transform(image_array).unsqueeze(0).to(self.device)
                    tensor_len = torch.tensor([target_length]).to(self.device)
                    with torch.no_grad():
                        output_emb = self.model(tensor_img, tensor_len).squeeze(0)
                        
                        # Cosine similarity for valid words
                        valid_words = filter_words_by_hint(self.vocab, target_hint)
                        if valid_words:
                            logits = []
                            for word in valid_words:
                                word_emb = self.word_embeddings[word]
                                sim = F.cosine_similarity(output_emb, word_emb, dim=0)
                                logits.append(sim)
                                
                            logits_tensor = torch.stack(logits)
                            # Apply softmax over the valid words. We multiply by 10.0 (temperature scaling)
                            # because cosine similarities are bounded [-1, 1], and a higher scale makes
                            # the softmax probabilities much more responsive and sharp.
                            probabilities = torch.softmax(logits_tensor * 10.0, dim=0).tolist()
                            
                            similarities = [(valid_words[i], probabilities[i]) for i in range(len(valid_words))]
                            similarities.sort(key=lambda x: x[1], reverse=True)
                            predictions = similarities[:3]
                else:
                    # Mock mode without torch
                    valid_words = filter_words_by_hint(self.vocab, target_hint)
                    if valid_words:
                        # Pick 3 random valid words
                        chosen = random.sample(valid_words, min(3, len(valid_words)))
                        probs = [random.uniform(0.1, 0.9) for _ in chosen]
                        probs.sort(reverse=True)
                        for word, p in zip(chosen, probs):
                            predictions.append((word, p))
                            
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break
                self.output_queue.put(predictions)
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Prediction error: {e}")
                
    def stop(self):
        self.running = False
