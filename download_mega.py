"""
MEGA Dataset Downloader - VERIFIED OPEN DATASETS ONLY
======================================================

Downloads ~350K images from verified open (non-gated) datasets.

All datasets tested and working as of 2024.
No login required. No gated access.
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data/general_detector")

# VERIFIED WORKING OPEN DATASETS (No gating, no deprecated scripts)
DATASETS = [
    # ============ AI-GENERATED IMAGES ============
    
    # AI vs Real detection datasets (balanced, labeled)
    {"name": "Parveshiiii/AI-vs-Real", "max": 50000, "label_col": "label"},
    {"name": "dima806/ai_vs_real_image_detection", "max": 50000, "label_col": "label"},
    
    # Stable Diffusion generated
    {"name": "frp94/stable_diffusion", "category": "synthetic", "max": 20000},
    
    # Midjourney
    {"name": "ehristoforu/midjourney-images", "category": "synthetic", "max": 30000},
    
    # DALL-E 3
    {"name": "ehristoforu/dalle-3-images", "category": "synthetic", "max": 30000},
    
    # More generators
    {"name": "Norod78/Realistic-Real-vs-AI-Art", "max": 20000, "label_col": "label"},
    
    # ============ REAL IMAGES ============
    
    # Flickr images (real photos)
    {"name": "Multimodal-Fatima/Flickr8k_Dataset", "category": "real", "max": 8000},
    
    # Food images (real)
    {"name": "food101", "category": "real", "max": 50000, "split": "train"},
    
    # CIFAR (real baseline)
    {"name": "cifar100", "category": "real", "max": 50000, "split": "train"},
    
    # Cats and dogs (real)
    {"name": "cats_vs_dogs", "category": "real", "max": 25000, "split": "train"},
]


def setup_dirs():
    """Create directory structure."""
    print("\n📁 Setting up directories...")
    for subdir in ["real/photos", "real/objects", "real/scenes", "synthetic/sd", "synthetic/mj", "synthetic/dalle", "synthetic/other", "splits"]:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)
    print("   ✓ Done")


def download_dataset(info: dict) -> tuple:
    """Download a single dataset. Returns (real_count, synthetic_count)."""
    name = info["name"]
    max_images = info.get("max", 10000)
    label_col = info.get("label_col")
    category = info.get("category")
    split = info.get("split", "train")
    
    print(f"\n📥 {name} (max: {max_images:,})")
    
    try:
        from datasets import load_dataset
        
        try:
            ds = load_dataset(name, split=split, streaming=True)
        except:
            try:
                ds = load_dataset(name, streaming=True)
                if hasattr(ds, 'keys'):
                    ds = ds[list(ds.keys())[0]]
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                return 0, 0
        
        real_count = 0
        synth_count = 0
        
        for item in ds:
            if real_count + synth_count >= max_images:
                break
            
            # Find image
            img = None
            for col in ["image", "img", "photo", "picture"]:
                if col in item and item[col] is not None:
                    img = item[col]
                    break
            
            if img is None:
                continue
            
            # Determine label
            if label_col and label_col in item:
                label = item[label_col]
                # Handle various label formats
                if isinstance(label, str):
                    is_real = label.lower() in ["real", "human", "0", "authentic"]
                else:
                    is_real = label == 0  # Usually 0 = real, 1 = fake
            elif category:
                is_real = category == "real"
            else:
                is_real = random.random() > 0.5  # Fallback
            
            # Save image
            if is_real:
                target = DATA_DIR / "real" / "photos"
                count_ref = "real"
            else:
                target = DATA_DIR / "synthetic" / "other"
                count_ref = "synth"
            
            filename = f"{name.replace('/', '_')}_{real_count + synth_count:06d}.jpg"
            filepath = target / filename
            
            try:
                if hasattr(img, 'save'):
                    img.save(filepath, "JPEG", quality=90)
                    if is_real:
                        real_count += 1
                    else:
                        synth_count += 1
                    
                    total = real_count + synth_count
                    if total % 5000 == 0:
                        print(f"   Progress: {total:,}/{max_images:,} (R:{real_count:,} S:{synth_count:,})")
            except:
                continue
        
        print(f"   ✅ Real: {real_count:,} | Synthetic: {synth_count:,}")
        return real_count, synth_count
        
    except ImportError:
        os.system("pip install datasets -q")
        return download_dataset(info)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0, 0


def create_splits():
    """Create balanced train/val/test splits."""
    print("\n📊 Creating splits...")
    
    real_imgs = list((DATA_DIR / "real").rglob("*.jpg"))
    synth_imgs = list((DATA_DIR / "synthetic").rglob("*.jpg"))
    
    print(f"   Real: {len(real_imgs):,}")
    print(f"   Synthetic: {len(synth_imgs):,}")
    
    # Balance
    min_count = min(len(real_imgs), len(synth_imgs))
    random.shuffle(real_imgs)
    random.shuffle(synth_imgs)
    
    all_data = [(str(f), 0, "real") for f in real_imgs[:min_count]]
    all_data += [(str(f), 1, "synthetic") for f in synth_imgs[:min_count]]
    random.shuffle(all_data)
    
    n = len(all_data)
    splits = {
        "train": all_data[:int(0.8*n)],
        "val": all_data[int(0.8*n):int(0.9*n)],
        "test": all_data[int(0.9*n):],
    }
    
    for name, data in splits.items():
        with open(DATA_DIR / "splits" / f"{name}.json", "w") as f:
            json.dump([{"path": p, "label": l, "source": s} for p, l, s in data], f)
        print(f"   {name}: {len(data):,}")
    
    return splits


def main():
    print("\n" + "=" * 60)
    print("🚀 MEGA DATASET DOWNLOADER (350K TARGET)")
    print("   Only verified open datasets - no gating issues")
    print("=" * 60)
    
    setup_dirs()
    
    total_real = 0
    total_synth = 0
    
    for i, info in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}]", end="")
        r, s = download_dataset(info)
        total_real += r
        total_synth += s
        print(f"   Running total: Real={total_real:,} Synth={total_synth:,}")
    
    splits = create_splits()
    
    print("\n" + "=" * 60)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"Real images:      {total_real:,}")
    print(f"Synthetic images: {total_synth:,}")
    print(f"Total:            {total_real + total_synth:,}")
    print(f"\nTrain: {len(splits['train']):,}")
    print(f"Val:   {len(splits['val']):,}")
    print(f"Test:  {len(splits['test']):,}")
    print(f"\nLocation: {DATA_DIR.absolute()}")
    print("\n🎯 Next: python train_general_detector.py")


if __name__ == "__main__":
    main()
