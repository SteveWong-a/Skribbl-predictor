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
        
    # Load from vocab.txt instead of hardcoding
    vocab_path = os.path.join(base_dir, "data", "vocab.txt")
    subset_categories = []
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            subset_categories = [line.strip() for line in f if line.strip()]
    
    if not subset_categories:
        print("vocab.txt not found or empty! Downloading fallback categories...")
        subset_categories = ["apple", "banana", "cat", "dog", "house"]
    
    for category in subset_categories:
        # Encode URL because some categories might have spaces e.g., 'alarm clock'
        filename = f"{category}.ndjson"
        url = BASE_URL + quote(filename)
        local_path = os.path.join(DATA_DIR, filename)
        
        download_file(url, local_path)

if __name__ == "__main__":
    main()
