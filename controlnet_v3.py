import torch
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np
import os

OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use one of your real training images as base
REAL_PHOTO = "/home/kenny/barber_ai/training_data/micro_geometric/10_designs/master_0056.png"

print("🔥 Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

print("🔥 Loading img2img pipeline...")
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
print("✅ Ready!")

# Load real photo as base
base = Image.open(REAL_PHOTO).convert("RGB").resize((1024, 1024))

# Make grid canny only in hair region
img = np.zeros((1024, 1024), dtype=np.uint8)
for i in range(200, 800, 70):
    cv2.line(img, (i, 50), (i, 450), 255, 3)
for j in range(50, 450, 70):
    cv2.line(img, (200, j), (800, j), 255, 3)
canny = Image.fromarray(img).convert("RGB")
canny.save(f"{OUTPUT_DIR}/canny_v3.png")

PROMPT = (
    "professional barbershop photo, african american male, "
    "side profile, low taper fade, sharp lineup, "
    "bold grid pattern shaved into hair, "
    "razor carved geometric lines, high contrast design, "
    "visible scalp in shaved areas, ultra realistic, DSLR photo"
)
NEGATIVE = "blurry, low quality, cartoon, 3d, tattoo, messy, distorted, painting"

print("🎨 Generating...")
image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    image=base,
    control_image=canny,
    controlnet_conditioning_scale=0.7,
    strength=0.6,
    num_inference_steps=35,
    guidance_scale=8.0,
).images[0]

image.save(f"{OUTPUT_DIR}/controlnet_v3.png")
print("✅ Saved: controlnet_v3.png")
print("🔥 DONE!")

# V3.1 — Bolder grid, higher contrast
print("🎨 Generating V3.1 - bolder lines...")

img2 = np.zeros((1024, 1024), dtype=np.uint8)
for i in range(200, 800, 55):
    cv2.line(img2, (i, 50), (i, 500), 255, 5)  # thicker lines
for j in range(50, 500, 55):
    cv2.line(img2, (200, j), (800, j), 255, 5)  # thicker lines
canny2 = Image.fromarray(img2).convert("RGB")

image2 = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    image=base,
    control_image=canny2,
    controlnet_conditioning_scale=0.95,  # stronger control
    strength=0.55,
    num_inference_steps=40,
    guidance_scale=10.0,  # stronger prompt
).images[0]

image2.save(f"{OUTPUT_DIR}/controlnet_v31.png")
print("✅ Saved: controlnet_v31.png")
