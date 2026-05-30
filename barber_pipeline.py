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
print("🔥 Loading SDXL Inpaint...")
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

print("✅ Model ready")

# -----------------------------
# LOAD IMAGE
# -----------------------------
base = Image.open(INPUT_IMAGE).convert("RGB").resize((1024, 1024))
base_np = np.array(base)

# -----------------------------
# SIMPLE HAIR REGION (NO AI)
# -----------------------------
# Dark pixels = hair (simple but effective for fades)

gray = cv2.cvtColor(base_np, cv2.COLOR_RGB2GRAY)

# threshold for dark hair
_, hair_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

# clean mask
kernel = np.ones((7,7), np.uint8)
hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# CREATE SHAVE DESIGN (THICK CHANNELS)
# -----------------------------
design = np.zeros((1024, 1024), dtype=np.uint8)

for i in range(250, 550, 90):
    pts = []
    for j in range(250, 650, 10):
        x = i + int(50 * np.sin(j / 100.0))
        y = j
        pts.append([x, y])

    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(design, [pts], False, 255, thickness=50)  # 🔥 thick = real shave

# -----------------------------
# COMBINE: ONLY SHAVE INSIDE HAIR
# -----------------------------
final_mask = cv2.bitwise_and(design, hair_mask)

mask_img = Image.fromarray(final_mask)

# -----------------------------
# PROMPT
# -----------------------------
PROMPT = (
    "professional barbershop photo, african american man, side profile, "
    "clean low skin fade haircut, "

    "real razor shaved hair design, visible scalp channels, "
    "hair removed cleanly, natural fade blending, "

    "realistic skin texture, natural lighting, DSLR photo"
)

NEGATIVE = (
    "lines drawn on top, overlay, fake, cartoon, blurry, "
    "tattoo, paint, unrealistic"
)

# -----------------------------
# GENERATE
# -----------------------------
print("💈 Generating...")

for i in range(3):
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=base,
        mask_image=mask_img,
        strength=0.9,                # 🔥 allow real change
        guidance_scale=6.5,
        num_inference_steps=35
    ).images[0]

    image = ImageEnhance.Contrast(image).enhance(1.2)

    save_path = f"{OUTPUT_DIR}/barber_result_{i:02d}.png"
    image.save(save_path)

    print(f"✅ Saved: {save_path}")

print("🔥 DONE")
