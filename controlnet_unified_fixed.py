#!/usr/bin/env python3
import torch
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
REAL_PHOTO = "/home/kenny/barber_ai/training_data/micro_geometric/10_designs/master_0056.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
print("🔥 Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

print("🔥 Loading SDXL...")
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# -----------------------------
# LOAD LORAS (STABLE LEVEL)
# -----------------------------
print("🔥 Loading LoRAs...")
pipe.load_lora_weights(
    "/home/kenny/barber_ai/lora_outputs/micro_geometric/micro_geo_lora.safetensors",
    adapter_name="micro"
)

pipe.load_lora_weights(
    "/home/kenny/barber_ai/lora_outputs/razor_designs/razor_designs_lora-step00003000.safetensors",
    adapter_name="design"
)

pipe.set_adapters(["micro", "design"], adapter_weights=[1.2, 1.5])

print("✅ Pipeline ready!")

# -----------------------------
# LOAD IMAGE
# -----------------------------
base = Image.open(REAL_PHOTO).convert("RGB").resize((1024, 1024))
base_np = np.array(base)

# -----------------------------
# HAIR MASK (CRITICAL FIX)
# -----------------------------
gray = cv2.cvtColor(base_np, cv2.COLOR_RGB2GRAY)

# detect dark hair
_, hair_mask = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)

# clean mask
kernel = np.ones((9,9), np.uint8)
hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# CONTROL PATTERN
# -----------------------------
img = np.zeros((1024, 1024), dtype=np.uint8)

center_x = 512
center_y = 540

for angle in range(40, 140, 4):
    rad = np.radians(angle)
    for r in range(180, 420, 10):
        x = int(center_x + r * np.cos(rad))
        y = int(center_y + r * np.sin(rad))

        if 0 <= x < 1024 and 0 <= y < 1024:
            cv2.circle(img, (x, y), 3, 255, -1)

# 🔥 APPLY HAIR MASK (THIS FIXES DISTORTION)
img = cv2.bitwise_and(img, hair_mask)

control_img = Image.fromarray(img).convert("RGB")

# -----------------------------
# PROMPT
# -----------------------------
PROMPT = (
    "real barber haircut, african american man, side profile, "
    "clean low fade, "

    "deep razor carved grooves inside hair only, "
    "visible scalp channels, strong contrast between dark hair and light scalp, "
    "sharp barber carving, grooves follow head shape, "

    "natural lighting, realistic skin texture, DSLR photo"
)

NEGATIVE = (
    "distorted face, melted skin, stretched head, "
    "overlay, drawing, abstract pattern, blurry, cartoon"
)

# -----------------------------
# GENERATE
# -----------------------------
print("🎨 Generating fixed results...")

for i in range(3):
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=base,
        control_image=control_img,

        controlnet_conditioning_scale=0.80,
        strength=0.55,
        guidance_scale=7.8,
        num_inference_steps=36
    ).images[0]

    # sharpen result
    image = ImageEnhance.Contrast(image).enhance(1.35)
    image = ImageEnhance.Sharpness(image).enhance(1.6)

    path = f"{OUTPUT_DIR}/fixed_clean_{i:02d}.png"
    image.save(path)

    print(f"✅ Saved: {path}")

print("🔥 DONE — CLEAN CONTROL OUTPUT")
