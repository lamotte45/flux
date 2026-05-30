import torch
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np
import os

OUTPUT_DIR = "/home/kenny/barber_ai/test_outputs/portrait_engrave"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_face_canny(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    face = img[0:int(h*0.6), int(w*0.1):int(w*0.9)]
    face = cv2.resize(face, (512, 512))
    edges = cv2.Canny(face, 30, 100)
    canvas = np.zeros((1024, 1024), dtype=np.uint8)
    canvas[200:712, 300:812] = edges
    return Image.fromarray(canvas).convert("RGB")

print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
).to("cuda")

print("Loading SDXL img2img...")
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

print("Loading LoRA...")
pipe.load_lora_weights(
    "/home/kenny/barber_ai/lora_outputs/razor_sdxl_v3/razor_sdxl_v3_000002500.safetensors"
)
pipe.fuse_lora(lora_scale=0.9)

base_images = [f for f in os.listdir("/home/kenny/barber_ai/test_outputs/final") if f.endswith(".png")]
if base_images:
    base_path = f"/home/kenny/barber_ai/test_outputs/final/{base_images[0]}"
else:
    base_path = "/home/kenny/barber_ai/kobe_ref.jpg"

base_image = Image.open(base_path).convert("RGB").resize((1024, 1024))
control_image = get_face_canny("/home/kenny/barber_ai/kobe_ref.jpg")
control_image.save(f"{OUTPUT_DIR}/canny_face.png")
print("Face canny saved")

PROMPT = (
    "RazorArtStyle, african american man, natural brown skin, "
    "side profile, Kobe Bryant portrait engraved into skin fade, "
    "face shaved into hair using dark stubble shading technique, "
    "photorealistic barbershop photo, plain grey background, barber cape"
)

NEGATIVE = (
    "tattoo, ink, blurry, cartoon, floating object, "
    "busy background, watermark, mannequin, 3d render, "
    "patterned clothing, full body, basketball court"
)

for i in range(4):
    print(f"Generating {i+1}/4...")
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=base_image,
        control_image=control_image,
        controlnet_conditioning_scale=0.9,
        strength=0.65,
        num_inference_steps=35,
        guidance_scale=7.5,
        width=1024,
        height=1024,
    ).images[0]
    save_path = f"{OUTPUT_DIR}/kobe_hair_{i:02d}.png"
    image.save(save_path)
    print(f"Saved {save_path}")

print(f"Done! Check {OUTPUT_DIR}")
