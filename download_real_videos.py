import os
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download

def main():
    dest_dir = Path("datasets/raw_real_videos")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("  Downloading UCF-101 Subset (REAL VIDEOS)")
    print("="*60)
    print("Downloading UCF101 subset from HuggingFace (sayakpaul/ucf101-subset)...")
    
    try:
        archive_path = hf_hub_download(
            repo_id="sayakpaul/ucf101-subset",
            repo_type="dataset",
            filename="UCF101_subset.tar.gz"
        )
        print(f"Downloaded to cache: {archive_path}")
        
        print("Extracting videos...")
        with tarfile.open(archive_path, "r:*") as tar:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=str(dest_dir))
            
        print(f"\n[+] Extraction complete! Videos are saved in: {dest_dir}")
        print("\nNext step: Run the following to extract frames and retrain the frame model:")
        print("    python video_processor.py --batch datasets/raw_real_videos --output datasets/processed/custom_real")
        print("    python train_video_model.py --phase-a")
        
    except ImportError:
        print("[!] Missing huggingface_hub. Please run: pip install huggingface_hub")
    except Exception as e:
        print(f"[!] Download failed: {e}")

if __name__ == "__main__":
    main()
