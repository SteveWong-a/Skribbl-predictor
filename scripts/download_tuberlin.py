import os
import requests
import zipfile

# The official TU Berlin sketch dataset
TU_BERLIN_URL = "http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
tu_berlin_dir = os.path.join(data_dir, "tu_berlin_data")
zip_path = os.path.join(data_dir, "tuberlin.zip")

def download_and_extract():
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Download
    if not os.path.exists(tu_berlin_dir):
        if not os.path.exists(zip_path):
            print(f"Downloading TU Berlin dataset from {TU_BERLIN_URL} (this is ~2.4GB, it might take a while)...")
            response = requests.get(TU_BERLIN_URL, stream=True)
            if response.status_code == 200:
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download complete.")
            else:
                print(f"Failed to download. Status code: {response.status_code}")
                return
        
        # 2. Extract
        print("Extracting TU Berlin zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # This extracts a folder named 'png' into data/
            zip_ref.extractall(data_dir)
            
        print("Extraction complete. Deleting zip file...")
        os.remove(zip_path)
        
        # Rename the extracted 'png' folder to 'tu_berlin_data'
        extracted_folder = os.path.join(data_dir, "png")
        if os.path.exists(extracted_folder):
            os.rename(extracted_folder, tu_berlin_dir)
            print(f"Dataset successfully prepared at: {tu_berlin_dir}")
    else:
        print(f"TU Berlin dataset already exists at {tu_berlin_dir}")

if __name__ == "__main__":
    download_and_extract()
