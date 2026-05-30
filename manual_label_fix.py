from ultralytics import YOLO
import os
from pathlib import Path

# Load model - using yolov8n as a reliable base for 'person' detection
model = YOLO('yolov8n.pt') 

img_dir = Path('/home/user/barber_ai/dataset/images/train')
lbl_dir = Path('/home/user/barber_ai/dataset/labels/train')
lbl_dir.mkdir(parents=True, exist_ok=True)

images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"🛠️ Attempting to manually label {len(images)} images...")

for img_name in images:
    img_path = img_dir / img_name
    results = model.predict(img_path, conf=0.2, device='cuda')
    
    # Manually extract the box and write the file
    boxes = results[0].boxes.xywhn.cpu().tolist() # Normalized coordinates
    classes = results[0].boxes.cls.cpu().tolist()
    
    if boxes:
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        with open(lbl_path, 'w') as f:
            for box, cls in zip(boxes, classes):
                # Class 0 is 'person'. We'll map it to 0 for 'head' in your YAML
                f.write(f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n")
        print(f"✅ Created: {lbl_path.name}")
    else:
        # If no person detected, we'll create an empty file so YOLO doesn't crash
        # This keeps the image count and label count in sync
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        open(lbl_path, 'a').close() 
        print(f"⚠️ Empty label for: {img_name} (No subject detected)")

print(f"\n🚀 DONE. Manual check: {len(os.listdir(lbl_dir))} labels created.")
