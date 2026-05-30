from ultralytics import YOLO
import os

model = YOLO('yolo11n.pt') 
paths = [
    ('/home/user/barber_ai/dataset/images/train', '/home/user/barber_ai/dataset/labels/train'),
    ('/home/user/barber_ai/dataset/images/val', '/home/user/barber_ai/dataset/labels/val')
]

for img_p, lbl_p in paths:
    os.makedirs(lbl_p, exist_ok=True)
    for img_file in os.listdir(img_p):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            results = model(os.path.join(img_p, img_file))
            for r in results:
                # Save as YOLO txt format
                r.save_txt(os.path.join(lbl_p, os.path.splitext(img_file)[0] + '.txt'))
print("✅ Labels created successfully for AI Beauty Concepts dataset!")
