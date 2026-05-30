from ultralytics import YOLO
import os

model = YOLO('yolo11n.pt') 
# Define exactly where images are
base_path = '/home/user/barber_ai/dataset'
sets = ['train', 'val']

for s in sets:
    img_dir = os.path.join(base_path, 'images', s)
    lbl_dir = os.path.join(base_path, 'labels', s)
    os.makedirs(lbl_dir, exist_ok=True)
    
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Checking {len(images)} images in {s}...")
    
    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        # Create label name by removing old extension and adding .txt
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(lbl_dir, lbl_name)
        
        results = model(img_path, conf=0.25)
        for r in results:
            r.save_txt(lbl_path)
            
print("✅ Labels synced. Run 'ls /home/user/barber_ai/dataset/labels/train' to verify files exist now.")
