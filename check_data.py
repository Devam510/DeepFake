import os

def check_datasets():
    base_dir = "datasets"
    if not os.path.exists(base_dir):
        print("❌ 'datasets' folder not found!")
        return
    
    print(f"Checking datasets in {base_dir}...\n")
    
    total_images = 0
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    for d in os.listdir(base_dir):
        path = os.path.join(base_dir, d)
        if os.path.isdir(path):
            count = 0
            for root, _, files in os.walk(path):
                count += sum(1 for f in files if os.path.splitext(f)[1].lower() in exts)
            
            print(f"📦 {d:<25} : {count:>8,} images")
            total_images += count
            
    print("-" * 45)
    print(f"🚀 TOTAL IMAGES             : {total_images:>8,}")

    if total_images > 100000:
        print("\n✅ READY FOR TRAINING!")
    else:
        print("\n⚠️ Need more data (aim for 100K+)")

if __name__ == "__main__":
    check_datasets()
