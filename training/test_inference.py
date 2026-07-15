import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Add parent directory to path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictor import SkribblPredictorModel

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main(image_path, target_length):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_path = os.path.join(base_dir, "vocab.txt")
    model_path = os.path.join(base_dir, "skribbl_model.pth")
    
    # Load Vocab & Mock Embeddings (in production, use real embeddings)
    vocab = load_vocab(vocab_path)
    torch.manual_seed(42)
    word_embeddings = {word: torch.randn(300).to(device) for word in vocab}
    
    # Load Model
    model = SkribblPredictorModel(embedding_dim=300).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded weights from {model_path}")
    else:
        print(f"WARNING: {model_path} not found. Running with random weights.")
    
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return
        
    tensor_img = transform(img).unsqueeze(0).to(device)
    tensor_len = torch.tensor([target_length]).to(device)
    
    # Inference
    print(f"\nRunning inference for a {target_length}-letter word...")
    with torch.no_grad():
        output_emb = model(tensor_img, tensor_len).squeeze(0)
        
        valid_words = [w for w in vocab if len(w) == target_length]
        similarities = []
        for word in valid_words:
            word_emb = word_embeddings[word]
            sim = F.cosine_similarity(output_emb, word_emb, dim=0).item()
            normalized_sim = (sim + 1.0) / 2.0
            similarities.append((word, normalized_sim))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 3 Predictions:")
        for i, (word, prob) in enumerate(similarities[:3]):
            print(f"{i+1}. {word} (Confidence: {prob*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_inference.py <path_to_image> <target_word_length>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    target_length = int(sys.argv[2])
    main(image_path, target_length)
