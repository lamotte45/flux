#!/usr/bin/env python3
import torch
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os

OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
REAL_PHOTO = "/home/kenny/barber_ai/training_data/micro_geometric/10_designs/master_0056.png"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔥 Loading Models & LoRAs...")
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

pipe.load_lora_weights("/home/kenny/barber_ai/lora_outputs/micro_geometric/micro_geo_lora.safetensors", adapter_name="micro")
pipe.load_lora_weights("/home/kenny/barber_ai/lora_outputs/razor_designs/razor_designs_lora-step00003000.safetensors", adapter_name="design")

# HEAVY WEIGHTS: Forcing the 'carved' look
pipe.set_adapters(["micro", "design"], adapter_weights=[1.35, 1.65])

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

base = Image.open(REAL_PHOTO).convert("RGB").resize((1024, 1024))

# --- IMPROVED GEOMETRY: THICKER CHANNELS ---
img = np.zeros((1024, 1024), dtype=np.uint8)
center_x, center_y = 512, 520 # Centered on the back of the head

# Draw thick lines instead of dots to force "channels"
for angle in range(60, 125, 15):
    rad = np.radians(angle)
    start_point = (center_x + int(150 * np.cos(rad)), center_y + int(150 * np.sin(rad)))
    end_point = (center_x + int(400 * np.cos(rad)), center_y + int(400 * np.sin(rad)))
    # Thickness 5-8 makes the AI treat it as a physical gap in hair
    cv2.line(img, start_point, end_point, 255, thickness=6) 

# Add a circle to anchor the design like the basketball look
cv2.circle(img, (center_x, center_y), 220, 255, thickness=5)

control_img = Image.fromarray(img).convert("RGB")

# --- PROMPT: FOCUS ON DEPTH AND SCALP ---
PROMPT = (
    "photorealistic back view of a fade haircut, "
    "deeply carved razor patterns, 3D grooves in the hair, "
    "pale scalp visible inside the shaved channels, "
    "sharp clean edges, high contrast between dark hair and skin, "
    "subsurface scattering on skin, professional barber photography, 8k"
)

NEGATIVE = (
    "glowing lines, neon, ghosting, stickers, floating lines, "
    "cartoon, messy hair, long hair, bad anatomy, flat design"
)

print("🎨 Generating Surgical Grade Designs...")

for i in range(4):
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=base,
        control_image=control_img,
        controlnet_conditioning_scale=1.1, # Pushed past 1.0 for dominance
        strength=0.72,                     # Higher strength to let AI 'carve' into the base
        num_inference_steps=40,            # More steps for finer texture
        guidance_scale=11.0                # Force the prompt's 'deeply carved' instruction
    ).images[0]

    # Post-process for "The Fresh Cut" Pop
    image = ImageEnhance.Contrast(image).enhance(1.25)
    image = ImageEnhance.Sharpness(image).enhance(1.5)

    path = f"{OUTPUT_DIR}/pro_carved_design_{i:02d}.png"
    image.save(path)
    print(f"✅ Saved: {path}")

print("🔥 DONE — Check the 'pro_carved' results")
