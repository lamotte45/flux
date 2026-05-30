import cv2
from ultralytics import YOLO
import os
from pathlib import Path

# Load your freshly trained 4090 model
model = YOLO('/home/user/runs/detect/train4/weights/best.pt')

raw_dir = Path('/home/user/barber_ai/raw_photos')
out_dir = Path('/home/user/barber_ai/training_data/smart_cropped')
out_dir.mkdir(parents=True, exist_ok=True)

print(f"✂️ Smart-cropping images from {raw_dir}...")

for img_path in raw_dir.glob('*.[jp][pn]g'):
    img = cv2.imread(str(img_path))
    h, w, _ = img.shape
    
    results = model(img)
    for r in results:
        if len(r.boxes) > 0:
            # Get the bounding box of the head/hair
            # xyxy format: [x1, y1, x2, y2]
            b = r.boxes.xyxy[0].cpu().numpy()
            
            # Add 15% padding to the top to prevent "chopping"
            pad = int((b[3] - b[1]) * 0.15)
            y1 = max(0, int(b[1] - pad))
            y2 = min(h, int(b[3] + pad))
            x1 = max(0, int(b[0] - pad))
            x2 = min(w, int(b[2] + pad))
            
            # Crop and resize to a clean 1024x1024
            crop = img[y1:y2, x1:x2]
            final = cv2.resize(crop, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
            
            out_name = out_dir / f"smart_{img_path.name}"
            cv2.imwrite(str(out_name), final)
            print(f"✅ Processed: {out_name.name}")

print(f"\n🚀 Done! Your 'Perfectly Framed' dataset is in {out_dir}")
