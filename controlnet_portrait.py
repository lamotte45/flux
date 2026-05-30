import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np
import os

OUTPUT_DIR = "/home/kenny/barber_ai/test_outputs/portrait_engrave"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def portrait_to_canny(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1024, 1024))
    edges = cv2.Canny(img, 50, 150)
    return Image.fromarray(edges).convert("RGB")

print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
).to("cuda")

print("Loading SDXL...")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

print("Loading LoRA...")
pipe.load_lora_weights(
    "/home/kenny/barber_ai/lora_outputs/razor_sdxl_v3/razor_sdxl_v3_000002500.safetensors"
)
pipe.fuse_lora(lora_scale=0.9)

control_image = portrait_to_canny("/home/kenny/barber_ai/kobe_ref.jpg")
control_image.save(f"{OUTPUT_DIR}/canny_preview.png")
print("Canny edges saved")

PROMPT = (
    "RazorArtStyle, african american man, natural brown skin, "
    "side profile, portrait engraved into skin fade, "
    "dark hair stubble creates shading, lighter stubble creates highlights, "
    "photorealistic barbershop photo, plain grey background, barber cape"
)

NEGATIVE = (
    "tattoo, ink, blurry, cartoon, metallic, floating object, "
    "busy background, text overlay, watermark, mannequin, plastic skin, "
    "3d render, patterned clothing"
)

for i in range(4):
    print(f"Generating {i+1}/4...")
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=control_image,
        controlnet_conditioning_scale=0.8,
        num_inference_steps=35,
        guidance_scale=7.5,
        width=1024,
        height=1024,
    ).images[0]
    image.save(f"{OUTPUT_DIR}/portrait_{i:02d}.png")
    print(f"Saved portrait_{i:02d}.png")

print(f"Done! Check {OUTPUT_DIR}")
