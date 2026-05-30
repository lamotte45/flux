#!/usr/bin/env python3
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_IMAGE = "/home/kenny/barber_ai/training_data/micro_geometric/10_designs/master_0056.png"
OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
print("🔥 Loading SDXL INPAINT...")
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

print("✅ Inpaint ready")

# -----------------------------
# LOAD IMAGE
# -----------------------------
base = Image.open(INPUT_IMAGE).convert("RGB").resize((1024, 1024))
base_np = np.array(base)

# -----------------------------
# STEP 1: HAIR MASK (DARK AREAS)
# -----------------------------
gray = cv2.cvtColor(base_np, cv2.COLOR_RGB2GRAY)

# detect hair
_, hair_mask = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((9,9), np.uint8)
hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# STEP 2: DESIGN (SHAVE CHANNELS)
# -----------------------------
design = np.zeros((1024, 1024), dtype=np.uint8)

center_x = 512
center_y = 540

for angle in range(40, 140, 5):
    rad = np.radians(angle)
    for r in range(200, 380, 12):
        x = int(center_x + r * np.cos(rad))
        y = int(center_y + r * np.sin(rad))

        if 0 <= x < 1024 and 0 <= y < 1024:
            cv2.circle(design, (x, y), 20, 255, -1)  # 🔥 THICK = real shave

# smooth edges
design = cv2.GaussianBlur(design, (21,21), 0)

# -----------------------------
# STEP 3: FINAL MASK (CRITICAL)
# -----------------------------
final_mask = cv2.bitwise_and(design, hair_mask)

mask_img = Image.fromarray(final_mask)

# -----------------------------
# PROMPT
# -----------------------------
PROMPT = (
    "professional barbershop photo, african american man, side profile, "
    "clean low skin fade haircut, "

    "deep shaved razor design, hair completely removed in channels, "
    "visible scalp, smooth skin inside cuts, "
    "strong contrast between dark hair and light shaved areas, "

    "realistic lighting, natural skin texture, DSLR photo"
)

NEGATIVE = (
    "lines drawn, overlay, fake, cartoon, blurry, "
    "tattoo, paint, unrealistic"
)

# -----------------------------
# GENERATE
# -----------------------------
print("💈 Generating REAL shave...")

for i in range(3):
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=base,
        mask_image=mask_img,

        strength=0.95,           # 🔥 FORCE REPLACEMENT
        guidance_scale=6.5,
        num_inference_steps=40
    ).images[0]

    image = ImageEnhance.Contrast(image).enhance(1.3)

    path = f"{OUTPUT_DIR}/inpaint_shave_{i:02d}.png"
    image.save(path)

    print(f"✅ Saved: {path}")

print("🔥 DONE — REAL SHAVE COMPLETE")
