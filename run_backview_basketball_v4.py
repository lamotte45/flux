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

# Correct adapter name is "default_0"
pipe.fuse_lora(lora_scale=1.2)
pipe.set_adapters(["default_0"], adapter_weights=[1.2])

print("✅ Ready")

# -------------------------------
# PROMPT (REALISTIC BACK VIEW)
# -------------------------------
PROMPT = """
real barbershop photo, african american boy, back of head view,
low skin fade haircut, short textured curls on top, sharp lineup,
EXTREMELY bold basketball design shaved deeply into the fade,
design carved with clean intersecting arcs forming a basketball pattern,
design centered on the back of the head, large and dominant,
razor lines crisp and high contrast, fade gradient smooth and realistic,
natural skin texture, visible pores, natural imperfections,
authentic barbershop background, neutral gray backdrop,
handheld camera feel, natural lighting, iPhone 15 photo quality,
shallow depth of field, <lora:barber_design_v2:1.4>
"""

NEGATIVE = """
cartoon, anime, illustration, vector, 3d render, plastic skin, waxy skin,
fake lighting, smooth skin, blurry fade, faint design, low contrast design,
studio render, jewelry, tattoos
"""

# -------------------------------
# GENERATE
# -------------------------------
print("🎨 Generating realistic backhead basketball fade...")

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    num_inference_steps=40,
    guidance_scale=6.0,
    width=1024,
    height=1024
).images[0]

# -------------------------------
# SAVE
# -------------------------------
output_path = f"{OUTPUT_DIR}/v4_backhead_basketball.png"
image.save(output_path)

print(f"✅ Saved: {output_path}")
