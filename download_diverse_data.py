"""
MEGA Dataset Downloader for General Image Detector
===================================================

Downloads comprehensive training data: ~385,000 images (~93.5 GB)

REAL IMAGES (~180,000):
- COCO (objects, scenes) - 50,000
- Flickr/Places (scenes) - 40,000  
- ImageNet subset - 40,000
- SUN Database - 30,000
- iNaturalist - 20,000

SYNTHETIC IMAGES (~205,000):
- Stable Diffusion 1.5 - 40,000
- Stable Diffusion 2.1 - 30,000
- Stable Diffusion XL - 30,000
- Midjourney v4/v5/v6 - 40,000
- DALL-E 2/3 - 30,000
- Firefly - 15,000
- Other - 20,000

Target: Production-quality detector with 97-99% accuracy.
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from datetime import datetime

# Configuration
DATA_DIR = Path("data/general_detector")

# MEGA Dataset Configuration
HUGGINGFACE_DATASETS = [
    # ==================== REAL IMAGES (~180,000) ====================
    
    # COCO - Objects and Scenes (50,000)
    {"name": "detection-datasets/coco", "category": "real", "domain": "objects", "max": 25000, "split": "train"},
    {"name": "detection-datasets/coco", "category": "real", "domain": "scenes", "max": 25000, "split": "val"},
    
    # Flickr - Diverse Scenes (40,000)
    {"name": "Multimodal-Fatima/Flickr8k_Dataset", "category": "real", "domain": "scenes", "max": 8000},
    {"name": "nlphuji/flickr30k", "category": "real", "domain": "scenes", "max": 30000},
    
    # ImageNet - Diverse Objects (40,000)
    {"name": "ILSVRC/imagenet-1k", "category": "real", "domain": "objects", "max": 40000, "split": "validation"},
    
    # Indoor/Outdoor (30,000)
    {"name": "keremberke/indoor-scene-classification", "category": "real", "domain": "indoor", "max": 15000},
    {"name": "Francesco/outdoor-scenes", "category": "real", "domain": "outdoor", "max": 15000},
    
    # Nature (20,000)
    {"name": "sasha/dog-food", "category": "real", "domain": "objects", "max": 10000},
    {"name": "keremberke/animal-classification", "category": "real", "domain": "objects", "max": 10000},
    
    # ==================== SYNTHETIC IMAGES (~205,000) ====================
    
    # Stable Diffusion 1.5 (40,000)
    {"name": "poloclub/diffusiondb", "category": "synthetic", "generator": "stable_diffusion", "max": 40000, "config": "2m_first_1k"},
    
    # Stable Diffusion 2.1 (30,000)
    {"name": "Falah/Handpick_Stable_Diffusion_V6", "category": "synthetic", "generator": "stable_diffusion_2", "max": 30000},
    
    # Stable Diffusion XL (30,000)
    {"name": "diffusers/benchmarks-datasets", "category": "synthetic", "generator": "stable_diffusion_xl", "max": 30000},
    
    # Midjourney v4/v5/v6 (40,000)
    {"name": "ehristoforu/midjourney-images", "category": "synthetic", "generator": "midjourney", "max": 20000},
    {"name": "Norod78/Midjourney-Showcase", "category": "synthetic", "generator": "midjourney", "max": 20000},
    
    # DALL-E 2/3 (30,000)
    {"name": "ehristoforu/dalle-3-images", "category": "synthetic", "generator": "dalle3", "max": 15000},
    {"name": "laion/dalle-3-dataset", "category": "synthetic", "generator": "dalle3", "max": 15000},
    
    # Firefly (15,000)
    {"name": "Norod78/Firefly-Samples", "category": "synthetic", "generator": "firefly", "max": 15000},
    
    # Other Generators (20,000)
    {"name": "wanng/midjourney-v5-202304-clean", "category": "synthetic", "generator": "other", "max": 10000},
    {"name": "OpenArtAI/openart-dataset", "category": "synthetic", "generator": "other", "max": 10000},
]


def setup_directories():
    """Create directory structure."""
    print("\n" + "=" * 60)
    print("SETTING UP DIRECTORIES")
    print("=" * 60)
    
    dirs = [
        DATA_DIR / "real" / "objects",
        DATA_DIR / "real" / "scenes", 
        DATA_DIR / "real" / "indoor",
        DATA_DIR / "real" / "outdoor",
        DATA_DIR / "real" / "faces",
        DATA_DIR / "synthetic" / "stable_diffusion",
        DATA_DIR / "synthetic" / "stable_diffusion_2",
        DATA_DIR / "synthetic" / "stable_diffusion_xl",
        DATA_DIR / "synthetic" / "midjourney",
        DATA_DIR / "synthetic" / "dalle3",
        DATA_DIR / "synthetic" / "firefly",
        DATA_DIR / "synthetic" / "other",
        DATA_DIR / "splits",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")


def copy_existing_data():
    """Copy existing data."""
    print("\n📋 Copying existing data...")
    
    sources = [
        ("data/prepared/synthetic_known", DATA_DIR / "synthetic" / "stylegan"),
        ("data/prepared/real_unverified", DATA_DIR / "real" / "faces"),
        ("data/research/by_generator/midjourney", DATA_DIR / "synthetic" / "midjourney"),
        ("data/research/by_generator/dalle3", DATA_DIR / "synthetic" / "dalle3"),
        ("data/research/real", DATA_DIR / "real" / "faces"),
    ]
    
    total = 0
    for src, dst in sources:
        src_path = Path(src)
        if src_path.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for f in src_path.glob("*.jpg"):
                if not (dst / f.name).exists():
                    shutil.copy2(f, dst / f.name)
                    total += 1
            print(f"   ✓ Copied from {src}")
    
    print(f"   Total copied: {total}")
    return total


def download_dataset(info: dict) -> int:
    """Download a single dataset."""
    name = info["name"]
    category = info["category"]
    max_images = info.get("max", 10000)
    
    if category == "synthetic":
        target = DATA_DIR / "synthetic" / info.get("generator", "other")
    else:
        target = DATA_DIR / "real" / info.get("domain", "objects")
    
    print(f"\n📥 {name}")
    print(f"   → {target} (max: {max_images:,})")
    
    try:
        from datasets import load_dataset
        
        config = info.get("config")
        split = info.get("split", "train")
        
        try:
            if config:
                ds = load_dataset(name, config, split=split, streaming=True, trust_remote_code=True)
            else:
                ds = load_dataset(name, split=split, streaming=True, trust_remote_code=True)
        except:
            try:
                ds = load_dataset(name, streaming=True, trust_remote_code=True)
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
            for col in ["image", "img", "photo", "picture", "jpg"]:
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


def create_splits():
    """Create train/val/test splits."""
    print("\n📊 Creating balanced splits...")
    
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
    
    # Balance
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


def create_metadata():
    """Create metadata."""
    meta = {"created": datetime.now().isoformat(), "real": {}, "synthetic": {}, "totals": {"real": 0, "synthetic": 0}}
    
    for d in (DATA_DIR / "real").iterdir():
        if d.is_dir():
            c = len(list(d.glob("*.jpg")))
            meta["real"][d.name] = c
            meta["totals"]["real"] += c
    
    for d in (DATA_DIR / "synthetic").iterdir():
        if d.is_dir():
            c = len(list(d.glob("*.jpg")))
            meta["synthetic"][d.name] = c
            meta["totals"]["synthetic"] += c
    
    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    return meta


def main():
    print("\n" + "=" * 60)
    print("🚀 MEGA DATASET DOWNLOADER")
    print("   Target: ~385,000 images (~93.5 GB)")
    print("=" * 60)
    
    setup_directories()
    copy_existing_data()
    
    print("\n🌐 Downloading from HuggingFace...")
    print("   This will take several hours...")
    
    total = 0
    for i, info in enumerate(HUGGINGFACE_DATASETS, 1):
        print(f"\n[{i}/{len(HUGGINGFACE_DATASETS)}]", end="")
        total += download_dataset(info)
    
    splits = create_splits()
    meta = create_metadata()
    
    print("\n" + "=" * 60)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"Real images:      {meta['totals']['real']:,}")
    print(f"Synthetic images: {meta['totals']['synthetic']:,}")
    print(f"Total:            {meta['totals']['real'] + meta['totals']['synthetic']:,}")
    print(f"\nTrain: {len(splits['train']):,}")
    print(f"Val:   {len(splits['val']):,}")
    print(f"Test:  {len(splits['test']):,}")
    print(f"\nLocation: {DATA_DIR.absolute()}")
    print("\n🎯 Next: python train_general_detector.py")


if __name__ == "__main__":
    main()
