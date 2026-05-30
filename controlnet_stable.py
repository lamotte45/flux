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
# OPTIONAL: LOAD LORAS (SAFE LEVEL)
# -----------------------------
try:
    print("🔥 Loading LoRAs...")
    pipe.load_lora_weights(
        "/home/kenny/barber_ai/lora_outputs/micro_geometric/micro_geo_lora.safetensors",
        adapter_name="micro"
    )
    pipe.load_lora_weights(
        "/home/kenny/barber_ai/lora_outputs/razor_designs/razor_designs_lora-step00003000.safetensors",
        adapter_name="design"
    )

    pipe.set_adapters(["micro", "design"], adapter_weights=[1.0, 1.3])
    print("✅ LoRAs loaded")
except:
    print("⚠️ Running without LoRA")

# -----------------------------
# LOAD IMAGE
# -----------------------------
base = Image.open(REAL_PHOTO).convert("RGB").resize((1024, 1024))

# -----------------------------
# HEAD-BOUND CONTROL PATTERN (FIXED)
# -----------------------------
img = np.zeros((1024, 1024), dtype=np.uint8)

for angle in range(30, 140, 8):
    rad = np.radians(angle)
    for r in range(220, 380, 12):
        x = int(480 + r * np.cos(rad))
        y = int(320 + r * np.sin(rad))

        # 🔥 HARD LIMIT TO HEAD AREA
        if 250 < x < 650 and 150 < y < 600:
            cv2.circle(img, (x, y), 2, 255, -1)

control_img = Image.fromarray(img).convert("RGB")

# -----------------------------
# PROMPT
# -----------------------------
PROMPT = (
    "real barber haircut, african american man, side profile, "
    "clean low skin fade, "

    "deep razor carved grooves inside fade only, "
    "visible scalp channels, high contrast between dark hair and light scalp, "
    "grooves follow head curvature, precise barber design, "

    "natural lighting, realistic skin texture, DSLR photo"
)

NEGATIVE = (
    "overlay, drawing, abstract, background pattern, distortion, "
    "blurry, cartoon, messy, stretched face"
)

# -----------------------------
# GENERATION
# -----------------------------
print("🎨 Generating controlled result...")

for i in range(3):
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=base,
        control_image=control_img,

        controlnet_conditioning_scale=0.75,   # 🔥 balanced
        strength=0.52,                        # 🔥 controlled edit
        guidance_scale=7.5,
        num_inference_steps=32
    ).images[0]

    image = ImageEnhance.Contrast(image).enhance(1.3)
    image = ImageEnhance.Sharpness(image).enhance(1.4)

    path = f"{OUTPUT_DIR}/stable_result_{i:02d}.png"
    image.save(path)

    print(f"✅ Saved: {path}")

print("🔥 DONE — stable results generated")
