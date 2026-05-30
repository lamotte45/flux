import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np
import os

OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔥 Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

print("🔥 Loading SDXL...")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

print("✅ Pipeline ready!")
print("🎨 Generating...")

def make_grid(size=1024):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, 80):
        cv2.line(img, (i, 0), (i, size), 255, 3)
        cv2.line(img, (0, i), (size, i), 255, 3)
    return Image.fromarray(img).convert("RGB")

PROMPT = "barber haircut photo, african american male, side profile, geometric pattern shaved into fade, razor carved design, high contrast shaved lines, ultra realistic, high detail"
NEGATIVE = "blurry, low quality, cartoon, 3d render, tattoo, messy, distorted"

canny = make_grid()
canny.save(f"{OUTPUT_DIR}/canny_grid.png")

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    image=canny,
    controlnet_conditioning_scale=0.9,
    num_inference_steps=30,
    guidance_scale=8.0,
    width=1024,
    height=1024,
).images[0]

image.save(f"{OUTPUT_DIR}/controlnet_grid.png")
print("✅ Saved: controlnet_grid.png")
print("🔥 DONE!")

def make_zigzag(size=1024):
    img = np.zeros((size, size), dtype=np.uint8)
    pts = [(i, 200 if (i//60)%2==0 else 400) for i in range(0, size, 60)]
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], 255, 4)
    return Image.fromarray(img).convert("RGB")

def make_diamond(size=1024):
    img = np.zeros((size, size), dtype=np.uint8)
    for x in range(100, size, 150):
        for y in range(100, size, 150):
            pts = np.array([[x,y-60],[x+60,y],[x,y+60],[x-60,y]], np.int32)
            cv2.polylines(img, [pts], True, 255, 3)
    return Image.fromarray(img).convert("RGB")

for name, canny in [("zigzag", make_zigzag()), ("diamond", make_diamond())]:
    print(f"➡️  {name}")
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=canny,
        controlnet_conditioning_scale=0.9,
        num_inference_steps=30,
        guidance_scale=8.0,
        width=1024,
        height=1024,
    ).images[0]
    image.save(f"{OUTPUT_DIR}/controlnet_{name}.png")
    print(f"✅ Saved: controlnet_{name}.png")
