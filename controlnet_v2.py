import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np
import os

OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔥 Loading ControlNet + SDXL...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
print("✅ Ready!")

def make_head_with_grid(size=1024):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = [40, 40, 40]
    # Head shape
    cv2.ellipse(img, (512, 400), (280, 340), 0, 0, 360, (180, 140, 100), -1)
    # Neck
    cv2.rectangle(img, (430, 680), (590, 820), (180, 140, 100), -1)
    # Grid ONLY in upper head/hair area
    for i in range(150, 700, 70):
        cv2.line(img, (i, 80), (i, 420), (255, 255, 255), 2)
    for j in range(80, 420, 70):
        cv2.line(img, (150, j), (700, j), (255, 255, 255), 2)
    # Canny edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return Image.fromarray(edges).convert("RGB")

canny = make_head_with_grid()
canny.save(f"{OUTPUT_DIR}/canny_head_grid.png")

PROMPT = (
    "professional barbershop photo, african american male, "
    "side profile, low taper fade, sharp lineup, "
    "bold grid pattern shaved into the hair, "
    "razor carved geometric lines in fade, "
    "high contrast design, visible scalp in shaved areas, "
    "natural lighting, ultra realistic, DSLR photo"
)
NEGATIVE = (
    "blurry, low quality, cartoon, 3d, "
    "tattoo, beard markings, messy, distorted, "
    "painting, illustration, watermark"
)

print("🎨 Generating head + grid...")
image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    image=canny,
    controlnet_conditioning_scale=0.8,
    num_inference_steps=35,
    guidance_scale=9.0,
    width=1024,
    height=1024,
).images[0]

image.save(f"{OUTPUT_DIR}/controlnet_head_grid.png")
print("✅ Saved: controlnet_head_grid.png")
print("🔥 DONE!")
