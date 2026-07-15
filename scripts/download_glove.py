import requests
import os
import zipfile
import sys

url = "http://nlp.stanford.edu/data/glove.6B.zip"
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

zip_path = os.path.join(data_dir, "glove.6B.zip")
txt_file = "glove.6B.300d.txt"
txt_path = os.path.join(data_dir, txt_file)

# Download with resume
headers = {}
if os.path.exists(zip_path):
    downloaded_bytes = os.path.getsize(zip_path)
    print(f"Resuming download from {downloaded_bytes} bytes...")
    headers['Range'] = f'bytes={downloaded_bytes}-'
else:
    downloaded_bytes = 0

try:
    with requests.get(url, headers=headers, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(zip_path, 'ab') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    sys.stdout.write(f"\rDownloaded {downloaded_bytes // (1024*1024)} MB")
                    sys.stdout.flush()
    print("\nDownload complete.")
except Exception as e:
    print(f"\nDownload failed: {e}")
    sys.exit(1)

print("\nExtracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extract(txt_file, path=data_dir)
print("Extraction complete. Deleting zip file...")
os.remove(zip_path)
print("Done!")
