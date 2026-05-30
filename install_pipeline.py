import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import os
import shutil
import random

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline

# ---------------- CONFIG ----------------
YOLO_WEIGHTS = "yolov8n.pt" 
SAM_CHECKPOINT = "/home/kenny/models/sam/sam_vit_h_4b8939.pth"
LORA_PATH = "/home/kenny/barber_ai/models/lora/ABC_KinkyStraight_v3.safetensors"
RAW_DIR = Path("/home/kenny/barber_ai/raw_photos")
OUT_DIR = Path("/home/kenny/barber_ai/outputs/install_pipeline")

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# VARIETY SETTINGS: The script will pick one for each model
STYLES = [
    "long flowing kinky straight hair, center part",
    "layered kinky straight hair, side sweep",
    "voluminous kinky straight hair, pushed back hairline",
    "sleek kinky straight hair, subtle honey brown highlights",
    "textured kinky straight hair, extra long length"
]

BASE_MODEL = "runwayml/stable-diffusion-inpainting"
device = "cuda"

# ---------------- LOAD ----------------
yolo = YOLO(YOLO_WEIGHTS)
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device)
sam_predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, safety_checker=None
).to(device)
pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=0.70)

# ---------------- PIPELINE ----------------
images = list(RAW_DIR.glob("*.png")) + list(RAW_DIR.glob("*.jpg"))

for i, img_path in enumerate(sorted(images)):
    # Pick a style from the list based on the image index
    current_style = STYLES[i % len(STYLES)]
    
    print(f"🖼 Style: {current_style} | Model: {img_path.name}")
    
    bgr = cv2.imread(str(img_path))
    if bgr is None: continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    results = yolo(rgb, verbose=False)
    if not results[0].boxes: continue
    
    x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    pad = int((x2 - x1) * 0.2)
    box = np.array([max(0, x1-pad), max(0, y1-pad), min(rgb.shape[1], x2+pad), y1 + int((y2-y1)*0.6)])

    sam_predictor.set_image(rgb)
    masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
    full_mask = (masks[np.argmax(scores)].astype(np.uint8) * 255)
    full_mask = cv2.GaussianBlur(full_mask, (25, 25), 0)

    init_img = Image.fromarray(cv2.resize(rgb, (1024,1024))).convert("RGB")
    mask_img = Image.fromarray(cv2.resize(full_mask, (1024,1024))).convert("L")

    # PROMPT CONSTRUCTION
    dynamic_prompt = f"photorealistic portrait of a woman, abcstyle hair, {current_style}, realistic skin pores, salon lighting"
    
    # REMOVE FIXED SEED: This is what creates the variety
    out_img = pipe(
        prompt=dynamic_prompt,
        negative_prompt="man, beard, jewelry, extra eyes, cartoon, cgi, flat top, messy",
        image=init_img, mask_image=mask_img,
        num_inference_steps=45, strength=0.62, guidance_scale=9.0
    ).images[0]

    out_img.save(OUT_DIR / f"variety_{i}_{img_path.stem}.png")

print(f"\n✅ Variety catalog generated in: {OUT_DIR}")
