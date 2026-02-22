"""Simple single dataset downloader"""
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

dataset_id = sys.argv[1] if len(sys.argv) > 1 else "philosopher0808/real-vs-ai-generated-faces-dataset"
output_dir = sys.argv[2] if len(sys.argv) > 2 else "datasets/real_vs_ai_faces"

print(f"Downloading {dataset_id} to {output_dir}...")

api = KaggleApi()
api.authenticate()

Path(output_dir).mkdir(parents=True, exist_ok=True)

api.dataset_download_files(
    dataset_id,
    path=output_dir,
    unzip=True,
    quiet=False
)

print(f"\n✅ Download complete: {output_dir}")

# Count images
from pathlib import Path
exts = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff'}
count = sum(1 for f in Path(output_dir).rglob('*') if f.suffix.lower() in exts)
print(f"   Images found: {count:,}")
