import os
from ultralytics import YOLO

# 1. Verify paths
img_dir = '/home/user/barber_ai/dataset/images/train'
lbl_dir = '/home/user/barber_ai/dataset/labels/train'
os.makedirs(lbl_dir, exist_ok=True)

print(f"Checking images in: {img_dir}")
files = os.listdir(img_dir)
print(f"Found {len(files)} files in directory.")

# 2. Load model
model = YOLO('yolo11n.pt')

# 3. Process
success_count = 0
for f in files:
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        full_path = os.path.join(img_dir, f)
        results = model(full_path, conf=0.1) # Ultra-low confidence to ensure we get something
        
        if len(results[0].boxes) > 0:
            # Save the labels
            label_file = os.path.splitext(f)[0] + '.txt'
            results[0].save_txt(os.path.join(lbl_dir, label_file))
            success_count += 1
        else:
            print(f"❓ Model couldn't find a subject in: {f}")

print(f"✅ Finished! Created {success_count} label files in {lbl_dir}")
