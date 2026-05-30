import torch
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os

# Configuration
OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
REAL_PHOTO = "/home/kenny/barber_ai/training_data/micro_geometric/10_designs/master_0056.png"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. LOAD MODELS
# -----------------------------
print("🔥 Loading Models...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# -----------------------------
# 2. HEAVY CONTROL IMAGE (THICKER LINES)
# -----------------------------
base = Image.open(REAL_PHOTO).convert("RGB").resize((1024, 1024))
img = np.zeros((1024, 1024), dtype=np.uint8)

# Thicker, more aggressive lines for ControlNet to "grab"
for i in range(250, 600, 80):
    points = []
    for j in range(250, 650, 10):
        x = i + int(60 * np.sin(j / 120.0))
        y = j
        points.append([x, y])
    
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    # Increased thickness to 6 for a "deep carve" look
    cv2.polylines(img, [pts], isClosed=False, color=255, thickness=6)

control_img = Image.fromarray(img).convert("RGB")

# -----------------------------
# 3. PROMPT & AGGRESSIVE SETTINGS
# -----------------------------
PROMPT = (
    "extreme barber shop photo, side profile, african american man, "
    "deeply shaved geometric hair design, razor carved lines, "
    "skin visible inside the hair design, high contrast hair art, "
    "crisp skin fade, cinematic lighting, 8k resolution, professional photography"
)

NEGATIVE = (
    "faint lines, ghost lines, tattoo, drawing, blurry, low res, "
    "bad hair, long hair, messy, grey hair, skin art, transparent"
)

print("🎨 Generating with HIGH CONTROL strength...")

for i in range(3):
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=base,
        control_image=control_img,
        # --- THE FIX ---
        controlnet_conditioning_scale=1.5,   # 🔥 Overdrive (Standard is 1.0)
        strength=0.75,                        # 🔥 Lowered "hold" on original image (more AI freedom)
        num_inference_steps=40,
        guidance_scale=12.0                   # 🔥 Higher guidance to force prompt adherence
    ).images[0]

    # Post-processing to pop the whites of the scalp
    image = ImageEnhance.Contrast(image).enhance(1.4)
    
    save_path = f"{OUTPUT_DIR}/high_strength_design_{i:02d}.png"
    image.save(save_path)
    print(f"✅ Saved: {save_path}")

print("\n🚀 COMPLETE: Check the 'high_strength' files!")
