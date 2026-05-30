import torch
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os

OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
REAL_PHOTO = "/home/kenny/barber_ai/training_data/micro_geometric/10_designs/master_0056.png"
os.makedirs(OUTPUT_DIR, exist_ok=True)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16, variant="fp16"
).to("cuda")

pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16, variant="fp16"
).to("cuda")

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

base = Image.open(REAL_PHOTO).convert("RGB").resize((1024, 1024))

img = np.zeros((1024, 1024), dtype=np.uint8)
for i in range(220, 520, 45):
    for j in range(260, 720, 45):
        x = i + int(30 * np.sin(j / 70.0))
        cv2.circle(img, (x, j), 2, 255, -1)
control_img = Image.fromarray(img).convert("RGB")

PROMPT = (
    "professional barbershop photo, african american man, "
    "side profile, clean low skin fade, sharp lineup, "
    "bold geometric razor design carved into fade, "
    "high contrast shaved channels, realistic skin, "
    "natural lighting, DSLR photo, sharp focus"
)
NEGATIVE = (
    "blurry, low quality, cartoon, 3d, distorted face, "
    "melting, deformed, bad anatomy, plastic skin, "
    "tattoo, overlay, messy, watermark"
)

print("🎨 Generating fixed version...")
for i in range(4):
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=base,
        control_image=control_img,
        controlnet_conditioning_scale=0.7,
        strength=0.35,        # ✅ lower = preserve real face
        num_inference_steps=40,
        guidance_scale=7.0,
    ).images[0]
    path = f"{OUTPUT_DIR}/fixed_{i:02d}.png"
    image.save(path)
    print(f"✅ Saved: {path}")

print("🔥 DONE!")
