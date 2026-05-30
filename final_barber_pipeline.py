import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT = "/home/kenny/barber_ai/training_data/micro_geometric/10_designs/master_0056.png"
OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔥 Loading SDXL Inpaint...")

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16
).to(DEVICE)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# -----------------------------
# LOAD IMAGE
# -----------------------------
base = Image.open(INPUT).convert("RGB").resize((1024, 1024))
base_np = np.array(base)

# -----------------------------
# HAIR MASK
# -----------------------------
gray = cv2.cvtColor(base_np, cv2.COLOR_RGB2GRAY)
_, hair_mask = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((9,9), np.uint8)
hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# DESIGN (YOU CONTROL THIS)
# -----------------------------
design = np.zeros((1024,1024), dtype=np.uint8)

center_x = 512
center_y = 540

for angle in range(40, 140, 5):
    rad = np.radians(angle)
    for r in range(200, 380, 12):
        x = int(center_x + r * np.cos(rad))
        y = int(center_y + r * np.sin(rad))
        cv2.circle(design, (x, y), 18, 255, -1)

# soften edges
design = cv2.GaussianBlur(design, (21,21), 0)

# -----------------------------
# FINAL MASK (CRITICAL)
# -----------------------------
mask = cv2.bitwise_and(design, hair_mask)
mask_img = Image.fromarray(mask)

# -----------------------------
# PROMPT (REAL SHAVE)
# -----------------------------
PROMPT = (
    "professional barbershop photo, african american man, side profile, "
    "clean low fade haircut, "

    "hair completely shaved in pattern, visible scalp, smooth skin, "
    "deep razor cut, strong contrast between hair and scalp, "

    "realistic lighting, DSLR photo"
)

NEGATIVE = "drawing, overlay, fake, blurry, cartoon, tattoo"

print("💈 Generating REAL haircut...")

for i in range(3):
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=base,
        mask_image=mask_img,
        strength=0.95,  # 🔥 FORCE CUT
        guidance_scale=6.5,
        num_inference_steps=40
    ).images[0]

    image = ImageEnhance.Contrast(image).enhance(1.3)

    path = f"{OUTPUT_DIR}/final_cut_{i}.png"
    image.save(path)

    print(f"✅ Saved: {path}")

print("🔥 DONE — REAL BARBER OUTPUT")
