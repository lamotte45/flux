import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import os
import shutil

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline

# ---------------- CONFIG ----------------
YOLO_WEIGHTS = "yolov8n.pt" 
SAM_CHECKPOINT = "/home/kenny/models/sam/sam_vit_h_4b8939.pth"
LORA_PATH = "/home/kenny/barber_ai/models/lora/ABC_KinkyStraight_v3.safetensors"

RAW_DIR = Path("/home/kenny/barber_ai/raw_photos")
OUT_DIR = Path("/home/kenny/barber_ai/outputs/style_pipeline")

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "runwayml/stable-diffusion-inpainting"
device = "cuda"

# ---------------- CURATED LOOKS ----------------
LOOKS = [
    {"style": "sleek kinky straight install", "color": "natural black"},
    {"style": "voluminous silk press", "color": "honey blonde highlights"},
    {"style": "layered kinky straight", "color": "copper auburn"},
    {"style": "bone straight install", "color": "burgundy red"},
    {"style": "long flowing kinky straight", "color": "platinum blonde"},
    {"style": "textured kinky straight", "color": "chocolate brown"}
]

LORA_STRENGTH = 0.75
HEIGHT = WIDTH = 1024

# ---------------- LOAD MODELS ----------------
yolo = YOLO(YOLO_WEIGHTS)
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device)
sam_predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, safety_checker=None
).to(device)
pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=LORA_STRENGTH)

# ---------------- PIPELINE ----------------
images = list(RAW_DIR.glob("*.png")) + list(RAW_DIR.glob("*.jpg"))

for img_path in images:
    print(f"\n🖼 Processing Model: {img_path.name}")
    bgr = cv2.imread(str(img_path))
    if bgr is None: continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    results = yolo(rgb, verbose=False)
    if not results[0].boxes: continue
    
    x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    pad = int((x2 - x1) * 0.25) # Slightly more padding for volume
    box = np.array([max(0, x1-pad), max(0, y1-pad), min(rgb.shape[1], x2+pad), y1 + int((y2-y1)*0.6)])

    sam_predictor.set_image(rgb)
    masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
    full_mask = (masks[np.argmax(scores)].astype(np.uint8) * 255)
    full_mask = cv2.GaussianBlur(full_mask, (25, 25), 0)

    init_img = Image.fromarray(cv2.resize(rgb, (WIDTH, HEIGHT))).convert("RGB")
    mask_img = Image.fromarray(cv2.resize(full_mask, (WIDTH, HEIGHT))).convert("L")

    for look in LOOKS:
        style = look['style']
        color = look['color']
        
        prompt = f"photorealistic portrait of a woman, abcstyle hair, {style}, {color}, highly detailed hair strands, salon lighting"
        negative = "man, male, beard, jewelry, extra face, distorted, cartoon, cinky, gold hair"

        print(f"  🎨 Applying Look: {style} ({color})")
        
        # FIXED: Variable definition added here
        result_image = pipe(
            prompt=prompt, 
            negative_prompt=negative,
            image=init_img, 
            mask_image=mask_img,
            num_inference_steps=40, 
            strength=0.65, 
            guidance_scale=10.0
        ).images[0]

        safe_name = style.replace(" ", "_")
        safe_color = color.replace(" ", "_")
        out_path = OUT_DIR / f"{img_path.stem}_{safe_name}_{safe_color}.png"
        
        result_image.save(out_path)
        print(f"  ✅ Saved: {out_path.name}")

print(f"\n✨ DONE! Variety catalog saved to: {OUT_DIR}")
