import os
import requests
from urllib.parse import quote

# Note: This is designed to run on a Cloud GPU instance with sufficient storage.
# The script will download the .ndjson files for a subset of the Quick, Draw! categories.

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(base_dir, "data", "quickdraw_data")
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"

def download_file(url, local_path):
    # Only download if it doesn't already exist
    if os.path.exists(local_path):
        print(f"Already exists: {local_path}")
        return
        
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Success.")
    else:
        print(f"Failed with status code: {response.status_code}")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Load from vocab.txt
    vocab_path = os.path.join(base_dir, "data", "vocab.txt")
    vocab_words = []
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab_words = [line.strip().lower() for line in f if line.strip()]
    
    # Fetch the official 345 QuickDraw categories from GitHub
    categories_url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
    try:
        r = requests.get(categories_url)
        official_categories = [c.strip().lower() for c in r.text.split('\n') if c.strip()]
    except:
        official_categories = ["apple", "banana", "cat", "dog", "house"] # fallback
        
    # Intersect
    subset_categories = [cat for cat in official_categories if not vocab_words or cat in vocab_words]
    
    if not subset_categories:
        print("No valid QuickDraw categories found in vocab.txt! Downloading fallback categories...")
        subset_categories = ["apple", "banana", "cat", "dog", "house"]
    
    print(f"Found {len(subset_categories)} valid QuickDraw categories to download...")
    for category in subset_categories:
        # Encode URL because some categories might have spaces e.g., 'alarm clock'
        filename = f"{category}.ndjson"
        url = BASE_URL + quote(filename)
        local_path = os.path.join(DATA_DIR, filename)
        
        download_file(url, local_path)

if __name__ == "__main__":
    main()
