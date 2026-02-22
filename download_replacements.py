"""
Replacement Dataset Downloader
==============================

Downloads replacement datasets for the ones that failed.

WORKING DATASETS (verified 2024):
- ImagiNet: 200K images (photos, paintings, faces, misc)
- InfImagine/FakeImageDataset: Fake image detection dataset
- frp94/stable_diffusion: SD images in parquet format
- nuriachandra/Deepfake-Eval-2024: 2024 deepfake benchmark
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data/general_detector")

# Verified working datasets (no trust_remote_code issues)
REPLACEMENT_DATASETS = [
    # Real vs Fake datasets (balanced)
    {"name": "InfImagine/FakeImageDataset", "category": "synthetic", "generator": "stable_diffusion", "max": 20000},
    {"name": "frp94/stable_diffusion", "category": "synthetic", "generator": "stable_diffusion", "max": 10000},
    
    # 2024 Deepfake evaluation
    {"name": "nuriachandra/Deepfake-Eval-2024", "category": "synthetic", "generator": "deepfake_2024", "max": 2000},
    
    # More Stable Diffusion
    {"name": "Gustavosta/Stable-Diffusion-Prompts", "category": "synthetic", "generator": "stable_diffusion", "max": 5000},
    
    # Real images - ImageNet style
    {"name": "ILSVRC/imagenet-1k", "category": "real", "domain": "objects", "max": 20000, "split": "validation"},
    
    # Real images - diverse scenes
    {"name": "scene_parse_150", "category": "real", "domain": "scenes", "max": 10000},
]


def download_dataset(info: dict) -> int:
    """Download a single dataset."""
    name = info["name"]
    category = info["category"]
    max_images = info.get("max", 10000)
    
    if category == "synthetic":
        target = DATA_DIR / "synthetic" / info.get("generator", "other")
    else:
        target = DATA_DIR / "real" / info.get("domain", "objects")
    
    target.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 {name}")
    print(f"   → {target} (max: {max_images:,})")
    
    try:
        from datasets import load_dataset
        
        config = info.get("config")
        split = info.get("split", "train")
        
        try:
            if config:
                ds = load_dataset(name, config, split=split, streaming=True)
            else:
                ds = load_dataset(name, split=split, streaming=True)
        except:
            try:
                ds = load_dataset(name, streaming=True)
                if hasattr(ds, 'keys'):
                    ds = ds[list(ds.keys())[0]]
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                return 0
        
        count = 0
        for item in ds:
            if count >= max_images:
                break
            
            img = None
            for col in ["image", "img", "photo", "picture", "jpg", "pixel_values"]:
                if col in item and item[col] is not None:
                    img = item[col]
                    break
            
            if img is None:
                continue
            
            filename = f"{name.replace('/', '_')}_{count:06d}.jpg"
            filepath = target / filename
            
            try:
                if hasattr(img, 'save'):
                    img.save(filepath, "JPEG", quality=90)
                    count += 1
                    
                    if count % 1000 == 0:
                        print(f"   Downloaded: {count:,}/{max_images:,}")
            except:
                continue
        
        print(f"   ✅ {count:,} images")
        return count
        
    except ImportError:
        os.system("pip install datasets -q")
        return download_dataset(info)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0


def update_splits():
    """Update train/val/test splits with new data."""
    import random
    
    print("\n📊 Updating balanced splits...")
    
    real_imgs = []
    for d in (DATA_DIR / "real").iterdir():
        if d.is_dir():
            real_imgs.extend([(str(f), 0, d.name) for f in d.glob("*.jpg")])
    
    synth_imgs = []
    for d in (DATA_DIR / "synthetic").iterdir():
        if d.is_dir():
            synth_imgs.extend([(str(f), 1, d.name) for f in d.glob("*.jpg")])
    
    print(f"   Real: {len(real_imgs):,}")
    print(f"   Synthetic: {len(synth_imgs):,}")
    
    min_count = min(len(real_imgs), len(synth_imgs))
    random.shuffle(real_imgs)
    random.shuffle(synth_imgs)
    
    all_imgs = real_imgs[:min_count] + synth_imgs[:min_count]
    random.shuffle(all_imgs)
    
    n = len(all_imgs)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    splits = {
        "train": all_imgs[:train_end],
        "val": all_imgs[train_end:val_end],
        "test": all_imgs[val_end:],
    }
    
    for name, data in splits.items():
        with open(DATA_DIR / "splits" / f"{name}.json", "w") as f:
            json.dump([{"path": p, "label": l, "source": s} for p, l, s in data], f)
        print(f"   {name}: {len(data):,}")
    
    return splits


def main():
    print("\n" + "=" * 60)
    print("🔄 REPLACEMENT DATASET DOWNLOADER")
    print("=" * 60)
    
    total = 0
    for i, info in enumerate(REPLACEMENT_DATASETS, 1):
        print(f"\n[{i}/{len(REPLACEMENT_DATASETS)}]", end="")
        total += download_dataset(info)
    
    splits = update_splits()
    
    # Count totals
    real_count = sum(1 for _ in (DATA_DIR / "real").rglob("*.jpg"))
    synth_count = sum(1 for _ in (DATA_DIR / "synthetic").rglob("*.jpg"))
    
    print("\n" + "=" * 60)
    print("✅ REPLACEMENT DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"New images added: {total:,}")
    print(f"Total Real: {real_count:,}")
    print(f"Total Synthetic: {synth_count:,}")
    print(f"Grand Total: {real_count + synth_count:,}")
    print(f"\nTrain: {len(splits['train']):,}")
    print(f"Val: {len(splits['val']):,}")
    print(f"Test: {len(splits['test']):,}")


if __name__ == "__main__":
    main()
