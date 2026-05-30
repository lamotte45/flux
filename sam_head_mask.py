import cv2
import torch
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import os

# ---- CONFIG ----
# Using official YOLOv8n (it will download automatically)
YOLO_WEIGHTS = "yolov8n.pt" 
SAM_CHECKPOINT = "/home/user/models/sam/sam_vit_h_4b8939.pth"
# Path to your 31 training images
RAW_DIR = Path("/home/user/barber_ai/training_data/lora_input/20_abcstyle")
MASK_DIR = Path("/home/user/barber_ai/masks/head_sam")
MASK_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Initializing YOLO & SAM on {device}...")
yolo = YOLO(YOLO_WEIGHTS)
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device)
predictor = SamPredictor(sam)

image_files = list(RAW_DIR.glob("*.png")) + list(RAW_DIR.glob("*.jpg"))
print(f"📸 Found {len(image_files)} images in {RAW_DIR}")

if not image_files:
    print("❌ ERROR: No images found. Check your RAW_DIR path.")
    exit()

for img_path in image_files:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"⚠️ Failed to read {img_path.name}")
        continue
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # YOLO: Finding the person/head
    results = yolo(img_rgb, verbose=False)
    
    if not results[0].boxes:
        print(f"❓ YOLO missed the subject in {img_path.name}")
        continue

    # Get the bounding box of the person
    b = results[0].boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = b.astype(int)
    
    # ADJUSTMENT: Focus on the top 60% of the detection (Head/Hair area)
    head_y2 = y1 + int((y2 - y1) * 0.6) 
    pad = int((x2 - x1) * 0.15) # 15% padding for voluminous hair
    
    predictor.set_image(img_rgb)
    input_box = np.array([
        max(0, x1-pad), 
        max(0, y1-pad), 
        min(img_rgb.shape[1], x2+pad), 
        head_y2
    ])

    # SAM generates the high-precision mask
    masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
    
    # Select best mask and convert to 0-255 grayscale
    mask = masks[np.argmax(scores)].astype(np.uint8) * 255
    
    # Apply a slight blur to the mask edges for natural hair blending
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Save the result
    out_path = MASK_DIR / f"{img_path.stem}_mask.png"
    cv2.imwrite(str(out_path), mask)
    print(f"✅ Masked: {img_path.name}")

print(f"\n✨ DONE! Your masks are ready in: {MASK_DIR}")
