import os
from ultralytics import YOLO
from pathlib import Path

model = YOLO('/home/user/runs/detect/train4/weights/best.pt')
img_dir = Path('/home/user/barber_ai/raw_photos')
best_gap = 0
best_img = ""

for img_p in img_dir.glob('*.[jp][pn]g'):
    results = model(img_p)
    for r in results:
        if len(r.boxes) > 0:
            top_y = r.boxes.xyxy[0][1].item()
            if top_y > best_gap:
                best_gap = top_y
                best_img = img_p.name

print(f"🌟 Best image for training is: {best_img} with {best_gap:.0f}px of top clearance.")
