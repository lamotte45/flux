from ultralytics import YOLO
import os
from pathlib import Path

# Load the model
model = YOLO('yolo11n.pt') 

# Define paths precisely
base_dir = Path('/home/user/barber_ai/dataset')
img_folders = ['images/train', 'images/val']

for folder in img_folders:
    img_path = base_dir / folder
    # Create the label folder counterpart (labels/train, labels/val)
    lbl_path = base_dir / folder.replace('images', 'labels')
    lbl_path.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Processing {folder} ---")
    
    # Get all images
    images = [f for f in os.listdir(img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"⚠️ No images found in {img_path}")
        continue

    for img_name in images:
        img_full_path = img_path / img_name
        results = model.predict(img_full_path, conf=0.2, device='cuda') # Use your 4090
        
        for r in results:
            if len(r.boxes) > 0:
                # Construct label filename
                label_file = lbl_path / (img_full_path.stem + '.txt')
                # Save results to the specific file
                r.save_txt(str(label_file))
                print(f"✅ Labeled: {img_name}")
            else:
                print(f"❌ No subject detected in: {img_name}")

print("\n🚀 All done! Labels should be in the labels/ folder now.")
