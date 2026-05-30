import torch
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os

# --- CONFIG ---
DEVICE = "cuda"
OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
BASE_IMAGE_PATH = "/home/kenny/barber_ai/training_data/micro_geometric/10_designs/master_0056.png"
LORA_PATH = "/home/kenny/barber_ai/lora_outputs/micro_geometric/micro_geo_lora.safetensors"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔥 Loading ControlNet + SDXL Inpaint...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to(DEVICE)

pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16"
).to(DEVICE)

print("🔥 Loading Your Micro-LoRA...")
pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=0.8)

# --- PREPARE BASES ---
base_img = Image.open(BASE_IMAGE_PATH).convert("RGB").resize((1024, 1024))
base_np = np.array(base_img)

# 1. Create Design Mask (Where the razor hits)
mask_np = np.zeros((1024, 1024), dtype=np.uint8)
# Let's draw a sharp geometric 'X' pattern
cv2.line(mask_np, (300, 200), (700, 600), 255, 30)
cv2.line(mask_np, (700, 200), (300, 600), 255, 30)
mask_img = Image.fromarray(mask_np)

# 2. Create Control Image (Canny)
# This forces the model to follow the 'X' edges exactly
control_img = Image.fromarray(mask_np).convert("RGB")

print("🎨 Rendering Razor Design...")
# Strength 0.4-0.6 is the "Sweet Spot"
# High enough to carve hair, low enough to keep the face real
result = pipe(
    prompt="professional barber razor design, carved hair pattern, visible scalp, sharp fade, realistic skin texture, dslr",
    negative_prompt="distorted face, melting, plastic skin, blurry, cartoon, tattoo",
    image=base_img,
    mask_image=mask_img,
    control_image=control_img,
    controlnet_conditioning_scale=0.9,
    strength=0.5,
    num_inference_steps=35,
    guidance_scale=8.5
).images[0]

# Post-Process for "Pop"
result = ImageEnhance.Contrast(result).enhance(1.2)
result = ImageEnhance.Sharpness(result).enhance(1.4)

save_path = os.path.join(OUTPUT_DIR, "pro_carving_output.png")
result.save(save_path)
print(f"✅ SUCCESS! View your result at: {save_path}")
