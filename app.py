import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import gradio as gr

# Setup paths and download GloVe if missing
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from core.predictor import SkribblPredictorModel, filter_words_by_hint
from core.embedding_utils import load_glove_embeddings

glove_path = os.path.join(base_dir, "data", "glove.6B.300d.txt")
if not os.path.exists(glove_path):
    print("Downloading GloVe embeddings... this might take a minute on Hugging Face Spaces.")
    # import download_glove script and run
    import subprocess
    subprocess.run([sys.executable, os.path.join(base_dir, "scripts", "download_glove.py")])

vocab_path = os.path.join(base_dir, "data", "vocab.txt")
with open(vocab_path, 'r') as f:
    vocab = [line.strip() for line in f if line.strip()]

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SkribblPredictorModel(embedding_dim=300).to(device)

model_path = os.path.join(base_dir, "weights", "skribbl_model.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded model weights.")
else:
    print("WARNING: Model weights not found!")
model.eval()

# Load Embeddings
cpu_embeddings = load_glove_embeddings(vocab, glove_path)
word_embeddings = {word: vec.to(device) for word, vec in cpu_embeddings.items()}

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image, hint):
    if image is None:
        return "Please draw something."
        
    # In Gradio, 'sketch' tool returns a dict with 'image' and 'mask', or just an image depending on gradio version.
    # We will assume it's a PIL Image or numpy array representing the drawing.
    if isinstance(image, dict):
        # Gradio 4.x sketchpad returns a dict
        if 'composite' in image:
            image_array = np.array(image['composite'])
        else:
            # Fallback
            image_array = np.array(image.get('image', image))
    else:
        image_array = np.array(image)
        
    # Preprocess image to match our training data
    # Skribbl is black strokes on white background.
    # Gradio sketchpad usually has transparent background with black strokes (or white background).
    img_pil = Image.fromarray(image_array)
    # Convert RGBA to RGB with white background
    if img_pil.mode == 'RGBA':
        background = Image.new('RGBA', img_pil.size, (255,255,255))
        img_pil = Image.alpha_composite(background, img_pil).convert('RGB')
    else:
        img_pil = img_pil.convert('RGB')
        
    # Convert to grayscale and threshold to match Pygame logic
    gray_img = img_pil.convert('L')
    gray_array = np.array(gray_img)
    thresholded = np.where(gray_array < 240, 0, 255).astype(np.uint8)
    canvas_array = np.stack((thresholded,)*3, axis=-1)

    hint = str(hint).strip().lower()
    if not hint:
        hint = "_____" # default 5
        
    target_length = len(hint)

    tensor_img = transform(canvas_array).unsqueeze(0).to(device)
    tensor_len = torch.tensor([target_length]).to(device)
    
    with torch.no_grad():
        output_emb = model(tensor_img, tensor_len).squeeze(0)
        
        valid_words = filter_words_by_hint(vocab, hint)
        if not valid_words:
            return "No words match that hint pattern!"
            
        logits = []
        for word in valid_words:
            word_emb = word_embeddings[word]
            sim = F.cosine_similarity(output_emb, word_emb, dim=0)
            logits.append(sim)
            
        logits_tensor = torch.stack(logits)
        probabilities = torch.softmax(logits_tensor * 10.0, dim=0).tolist()
        
        similarities = [(valid_words[i], probabilities[i]) for i in range(len(valid_words))]
        similarities.sort(key=lambda x: x[1], reverse=True)
        predictions = similarities[:10]
        
    output_text = "### Top Predictions\n"
    for i, (word, prob) in enumerate(predictions):
        output_text += f"{i+1}. **{word}** ({prob*100:.1f}%)\n"
        
    return output_text

# Create Gradio Interface
with gr.Blocks(title="Skribbl AI Predictor") as demo:
    gr.Markdown("# 🎨 Skribbl AI Predictor")
    gr.Markdown("Draw your sketch below and type in the hint pattern (e.g. `__i_k__` or `_____`). The AI will filter its vocabulary and guess the word!")
    
    with gr.Row():
        with gr.Column():
            canvas = gr.Sketchpad(type="pil", label="Drawing Canvas")
            hint_input = gr.Textbox(label="Word Hint Pattern", placeholder="e.g. __i_k__ or 5 for five underscores")
            predict_btn = gr.Button("Predict", variant="primary")
            
        with gr.Column():
            output = gr.Markdown(label="Predictions")
            
    # Whenever the hint changes or the user clicks predict, update
    predict_btn.click(fn=predict, inputs=[canvas, hint_input], outputs=output)
    hint_input.change(fn=predict, inputs=[canvas, hint_input], outputs=output)

if __name__ == "__main__":
    demo.launch()
