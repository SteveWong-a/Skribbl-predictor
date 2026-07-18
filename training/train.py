import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchvision.transforms as transforms
from tqdm import tqdm
try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

# Add parent directory to path so we can import the model from predictor.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd()) # Fallback for Colab execution

from core.predictor import SkribblPredictorModel
from dataset import SkribblDataset, TUBerlinDataset
from core.embedding_utils import load_glove_embeddings

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    # 2. Set up the argument parser
    parser = argparse.ArgumentParser(description="Train Skribbl AI Predictor")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser.add_argument('--data_dir', type=str, default=os.path.join(base_dir, "data", "quickdraw_data"), 
                        help='Path to the QuickDraw dataset directory')
    parser.add_argument('--tu_berlin_dir', type=str, default=os.path.join(base_dir, "data", "tu_berlin_data"), 
                        help='Path to the TU Berlin dataset directory')
    parser.add_argument('--output_dir', type=str, default=os.path.join(base_dir, "weights"), 
                        help='Directory to save the trained model weights')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hf_repo_id', type=str, default=None, help='Optional: Hugging Face repo ID to upload model to (e.g. SteveaWong/AI-Drawing-Predictor)')
    
    # Use parse_known_args so Jupyter internal args don't crash the script
    args, _ = parser.parse_known_args()

    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_path = os.path.join(base_dir, "data", "vocab.txt")
    data_dir = args.data_dir
    tu_berlin_dir = args.tu_berlin_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    hf_repo_id = args.hf_repo_id

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using device: {device}")
    vocab = load_vocab(vocab_path)
    print(f"Loaded {len(vocab)} vocab words.")

    # Load real embeddings
    glove_path = os.path.join(base_dir, "data", "glove.6B.300d.txt")
    word_embeddings = load_glove_embeddings(vocab, glove_path)

    # 2. Data Loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    qd_dataset = SkribblDataset(data_dir=data_dir, vocab=vocab, embeddings=word_embeddings, transform=transform, max_samples_per_class=1000)
    tu_dataset = TUBerlinDataset(data_dir=tu_berlin_dir, vocab=vocab, embeddings=word_embeddings, transform=transform)
    
    datasets_to_concat = []
    if len(qd_dataset) > 0:
        datasets_to_concat.append(qd_dataset)
        print(f"Loaded {len(qd_dataset)} samples from QuickDraw (Progressive).")
    if len(tu_dataset) > 0:
        datasets_to_concat.append(tu_dataset)
        print(f"Loaded {len(tu_dataset)} samples from TU Berlin (100% Complete).")
        
    if not datasets_to_concat:
        print("No data found! Please ensure datasets are downloaded.")
        return
        
    full_dataset = ConcatDataset(datasets_to_concat)
    print(f"Total merged samples loaded: {len(full_dataset)}")
    
    # Train/Val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Model & Loss
    model = SkribblPredictorModel(embedding_dim=300).to(device)
    
    # CosineEmbeddingLoss compares two vectors and pushes them together (y=1) or apart (y=-1)
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, lengths, target_embeddings) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = images.to(device)
            lengths = lengths.to(device)
            target_embeddings = target_embeddings.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, lengths)
            
            # Loss: target tensor y=1 implies we want cosine similarity between outputs and target_embeddings to be 1
            y = torch.ones(outputs.size(0)).to(device)
            loss = criterion(outputs, target_embeddings, y)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, lengths, target_embeddings in val_loader:
                images = images.to(device)
                lengths = lengths.to(device)
                target_embeddings = target_embeddings.to(device)
                
                outputs = model(images, lengths)
                y = torch.ones(outputs.size(0)).to(device)
                loss = criterion(outputs, target_embeddings, y)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, "skribbl_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
            
            # Optionally upload to Hugging Face during training to avoid losing progress if the notebook restarts
            if hf_repo_id:
                if HfApi is None:
                    print("huggingface_hub is not installed! Cannot upload model. Run: pip install huggingface_hub")
                else:
                    if os.path.exists(model_path):
                        print(f"Uploading checkpoint to Hugging Face repo: {hf_repo_id}...")
                        try:
                            api = HfApi()
                            repo_type = "space" if "Predictor" in hf_repo_id else "model"
                            api.upload_file(
                                path_or_fileobj=model_path,
                                path_in_repo="weights/skribbl_model.pth",
                                repo_id=hf_repo_id,
                                repo_type=repo_type
                            )
                            print("✅ Successfully backed up checkpoint to Hugging Face!")
                        except Exception as e:
                            print(f"❌ Failed to upload checkpoint: {e}")

if __name__ == '__main__':
    main()
