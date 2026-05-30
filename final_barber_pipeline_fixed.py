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

print("Loading SDXL Inpaint...")

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

# -----------------------------
# FORCE FADE REGION
# -----------------------------
fade_mask = np.zeros((1024, 1024), dtype=np.uint8)
cv2.rectangle(fade_mask, (150, 250), (750, 700), 255, -1)

# -----------------------------
# DESIGN PATTERN
# -----------------------------
design = np.zeros((1024,1024), dtype=np.uint8)

center_x = 512
center_y = 540

for angle in range(40, 140, 5):
    rad = np.radians(angle)
    for r in range(200, 380, 12):
        x = int(center_x + r * np.cos(rad))
        y = int(center_y + r * np.sin(rad))
        if 0 <= x < 1024 and 0 <= y < 1024:
            cv2.circle(design, (x, y), 18, 255, -1)

design = cv2.GaussianBlur(design, (21,21), 0)

# -----------------------------
# FINAL MASK
# -----------------------------
final_mask = cv2.bitwise_and(fade_mask, design)
mask_img = Image.fromarray(final_mask)

# Save mask for debug
mask_img.save(os.path.join(OUTPUT_DIR, "debug_mask.png"))

# -----------------------------
# PROMPT
# -----------------------------
PROMPT = (
    "professional barbershop photo, african american man, side profile, "
    "clean low fade haircut, hair shaved into pattern, visible scalp, "
    "deep razor cut, strong contrast, realistic lighting"
)

NEGATIVE = "drawing, overlay, tattoo, blurry, cartoon"

print("Generating haircut...")

for i in range(3):
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=base,
        mask_image=mask_img,
        strength=1.0,
        guidance_scale=6.5,
        num_inference_steps=40
    ).images[0]

    image = ImageEnhance.Contrast(image).enhance(1.35)

    path = os.path.join(OUTPUT_DIR, f"final_cut_fixed_{i}.png")
    image.save(path)

    print("Saved:", path)

print("DONE")
