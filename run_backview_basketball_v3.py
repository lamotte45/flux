#!/usr/bin/env python3

import torch
from diffusers import StableDiffusionXLPipeline
import os

# -------------------------------
# CONFIG
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH = "/home/kenny/barber_ai/models/lora/barber_design_v2-step00001500.safetensors"

OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
print("🔥 Loading SDXL...")

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    use_safetensors=True
).to(DEVICE)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

print("🔥 Loading LoRA...")
pipe.load_lora_weights(LORA_PATH)

# 🔥 FORCE LORA TO DOMINATE
pipe.fuse_lora(lora_scale=1.8)
pipe.set_adapters(["default"], adapter_weights=[1.8])

print("✅ Ready")

# -------------------------------
# PROMPT (FORCED DESIGN)
# -------------------------------
PROMPT = """
RAW photo, studio headshot, african american male, back of head view,

low skin fade haircut, smooth short hair on top, no curls,

extremely bold shaved design carved deeply into the fade, impossible to miss,

circular shaved region with intersecting arcs forming basketball structure,

design dominates the haircut completely,

plain neutral background, studio lighting,
shallow depth of field, ultra realistic, high detail
"""

NEGATIVE = """
blurry, low contrast design, cartoon, anime, illustration, vector,
plastic skin, waxy skin, fake lighting
"""

# -------------------------------
# GENERATE
# -------------------------------
print("🎨 Generating...")

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    num_inference_steps=30,
    guidance_scale=5.5,
    width=1024,
    height=1024
).images[0]

# -------------------------------
# SAVE
# -------------------------------
output_path = f"{OUTPUT_DIR}/v3_force_design.png"
image.save(output_path)

print(f"✅ Saved: {output_path}")
