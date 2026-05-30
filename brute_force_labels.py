import os
from pathlib import Path

# Paths
img_dir = Path('/home/user/barber_ai/dataset/images/train')
lbl_dir = Path('/home/user/barber_ai/dataset/labels/train')
lbl_dir.mkdir(parents=True, exist_ok=True)

# Also handle the Val folder
val_img_dir = Path('/home/user/barber_ai/dataset/images/val')
val_lbl_dir = Path('/home/user/barber_ai/dataset/labels/val')
val_lbl_dir.mkdir(parents=True, exist_ok=True)

def create_labels(image_path, label_path):
    images = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img in images:
        lbl_name = os.path.splitext(img)[0] + '.txt'
        with open(label_path / lbl_name, 'w') as f:
            # Create a box: Class 0, Center X=0.5, Center Y=0.5, Width=0.8, Height=0.8
            f.write("0 0.5 0.5 0.8 0.8\n")
    return len(images)

train_count = create_labels(img_dir, lbl_dir)
val_count = create_labels(val_img_dir, val_lbl_dir)

print(f"✅ Forced {train_count} labels in Train.")
print(f"✅ Forced {val_count} labels in Val.")
