import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms as transforms


class TUBerlinDataset(Dataset):
    def __init__(self, data_dir, vocab, transform=None):
        self.data_dir = data_dir
        self.vocab = vocab
        self.transform = transform
        self.samples = []
        
        # Use the exact same embedding dictionary setup as your SkribblDataset
        torch.manual_seed(42)
        self.embeddings = {word: torch.randn(300) for word in vocab}
        
        # Walk through TU Berlin folders (each folder name is a category)
        if os.path.exists(data_dir):
            for category in os.listdir(data_dir):
                if category in self.vocab:
                    cat_dir = os.path.join(data_dir, category)
                    # Ensure it is a directory and not a hidden file like .DS_Store
                    if os.path.isdir(cat_dir):
                        for img_name in os.listdir(cat_dir):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.samples.append((os.path.join(cat_dir, img_name), category))

    def __getitem__(self, idx):
        img_path, word = self.samples[idx]
        
        # Load the static PNG image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, len(word), self.embeddings[word]

    def __len__(self):
        return len(self.samples)



class SkribblDataset(Dataset):
    def __init__(self, data_dir, vocab, transform=None, max_samples_per_class=1000):
        self.data_dir = data_dir
        self.vocab = vocab
        self.transform = transform
        self.samples = []
        
        # Load precomputed embeddings (in production, load GloVe/FastText)
        # Here we mock them with fixed random vectors for consistency
        torch.manual_seed(42)
        self.embeddings = {word: torch.randn(300) for word in vocab}
        
        self._load_data(max_samples_per_class)
        
    def _load_data(self, max_samples_per_class):
        # Scan through the downloaded .ndjson files
        if not os.path.exists(self.data_dir):
            return
            
        for file in os.listdir(self.data_dir):
            if file.endswith('.ndjson'):
                word = file.replace('.ndjson', '')
                if word not in self.vocab:
                    continue
                    
                path = os.path.join(self.data_dir, file)
                count = 0
                with open(path, 'r') as f:
                    for line in f:
                        if count >= max_samples_per_class:
                            break
                        try:
                            sample = json.loads(line)
                            # Only keep recognized samples
                            if sample.get('recognized', False):
                                self.samples.append((sample['drawing'], word))
                                count += 1
                        except json.JSONDecodeError:
                            continue
                            
    def __len__(self):
        return len(self.samples)
        
    def _draw_strokes_to_image(self, strokes):
        # Quick Draw coordinate system is [0, 255]
        # Random background color (mostly white/off-white)
        bg_color = (
            random.randint(240, 255),
            random.randint(240, 255),
            random.randint(240, 255)
        )
        img = Image.new('RGB', (256, 256), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Progressive Stroke Rendering / Sub-stroke masking
        # Focused on progressive strokes since TU Berlin dataset provides 100% complete drawings
        milestone = random.choice([0.25, 0.5, 0.75])
        if milestone < 1.0:
            total_points = sum(len(s[0]) for s in strokes)
            target_points = max(1, int(total_points * milestone))
            
            truncated_strokes = []
            points_so_far = 0
            for s in strokes:
                x_coords, y_coords = s[0], s[1]
                n = len(x_coords)
                
                if points_so_far + n <= target_points:
                    truncated_strokes.append([x_coords, y_coords])
                    points_so_far += n
                else:
                    remaining = target_points - points_so_far
                    if remaining > 0:
                        truncated_strokes.append([x_coords[:remaining], y_coords[:remaining]])
                    break
            strokes = truncated_strokes
            
        # Random stroke color for robustness
        if random.random() > 0.5:
            # 50% chance of standard black/dark stroke
            stroke_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
        else:
            # 50% chance of random colored stroke (keeping it somewhat dark to contrast with background)
            stroke_color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
            
        for stroke in strokes:
            x_coords = stroke[0]
            y_coords = stroke[1]
            # Connect the points in the stroke
            for i in range(len(x_coords) - 1):
                pt1 = (x_coords[i], y_coords[i])
                pt2 = (x_coords[i+1], y_coords[i+1])
                # Randomize thickness slightly for augmentation
                thickness = random.randint(2, 8)
                draw.line([pt1, pt2], fill=stroke_color, width=thickness)
                
        return img
        
    def __getitem__(self, idx):
        strokes, word = self.samples[idx]
        
        img = self._draw_strokes_to_image(strokes)
        
        if self.transform:
            img = self.transform(img)
            
        target_length = len(word)
        target_embedding = self.embeddings[word]
        
        return img, target_length, target_embedding
