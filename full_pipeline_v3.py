import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import os

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline

# ---------------- CONFIG ----------------
# Using official weights since custom YOLO training was incomplete
YOLO_WEIGHTS = "yolov8n.pt" 
SAM_CHECKPOINT = "/home/user/models/sam/sam_vit_h_4b8939.pth"
LORA_PATH = "/home/user/barber_ai/models/lora/ABC_KinkyStraight_v3.safetensors"

# Pointing to your input folder
RAW_DIR = Path("/home/user/barber_ai/training_data/lora_input/20_abcstyle")
OUT_DIR = Path("/home/user/barber_ai/outputs/install_pipeline_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "runwayml/stable-diffusion-inpainting" # Specific inpainting base
device = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = (
    "photorealistic studio portrait, abcstyle hair, long kinky straight hair texture, "
    "massive hair volume, rounded top, highly detailed strands, 8k, salon lighting"
)

NEGATIVE_PROMPT = (
    "extra face, teeth in hair, jewelry, distorted face, blurry, lowres, "
    "bad anatomy, cloned head, flat top, boxy hair, cartoon, skin artifacts"
)

LORA_STRENGTH = 0.75
HEIGHT = WIDTH = 1024 

# ---------------- LOAD MODELS ----------------
print("🚀 Loading AI Models into VRAM...")
yolo = YOLO(YOLO_WEIGHTS)
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device)
sam_predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, safety_checker=None
).to(device)
pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=LORA_STRENGTH)

# ---------------- PIPELINE ----------------
images = sorted(list(RAW_DIR.glob("*.png")) + list(RAW_DIR.glob("*.jpg")))
print(f"📸 Found {len(images)} images. Starting full install...")

for img_path in images:
    print(f"\n🖼 Processing: {img_path.name}")
    bgr = cv2.imread(str(img_path))
    if bgr is None: continue

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ---- YOLO: detect head box ----
    results = yolo(rgb, verbose=False)
    if not results[0].boxes:
        print("  ⚠️ No subject detected, skipping.")
        continue

    x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)

    # INCREASED PADDING FOR HAIR VOLUME
    pad = int((x2 - x1) * 0.25)
    box = np.array([max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), y1 + int((y2-y1)*0.6)])

    # ---- SAM: segment ----
    sam_predictor.set_image(rgb)
    masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
    full_mask = (masks[np.argmax(scores)].astype(np.uint8) * 255)

    # SOFTEN MASK EDGES (Crucial for realism)
    full_mask = cv2.GaussianBlur(full_mask, (15, 15), 0)

    # ---- Resize to LoRA resolution ----
    rgb_res = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    mask_res = cv2.resize(full_mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

    init_image = Image.fromarray(rgb_res).convert("RGB")
    mask_image = Image.fromarray(mask_res).convert("L")

    # ---- Inpaint ----
    generator = torch.Generator(device=device).manual_seed(1234)
    print("  🎨 Installing Hair...")
    out_img = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=40,
        strength=0.85,
        guidance_scale=11.0, # High guidance to kill artifacts
        generator=generator,
    ).images[0]

    out_path = OUT_DIR / f"{img_path.stem}_v3_final.png"
    out_img.save(out_path)
    print(f"  ✅ Saved: {out_path}")

print(f"\n✨ DONE! Full catalog saved to: {OUT_DIR}")
