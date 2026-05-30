import torch
from diffusers import StableDiffusionXLPipeline
import os

OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔥 Loading SDXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

print("✅ Model ready — generating test image...")

prompt = "carved hair design, shaved line pattern, razor sharp lines, precise hair carving, geometric pattern, real barbershop photo, african american male, side profile, ultra realistic"

image = pipe(
    prompt=prompt,
    negative_prompt="blurry, low quality, cartoon, 3d render",
    num_inference_steps=25,
    guidance_scale=7.5,
    width=1024,
    height=1024,
).images[0]

image.save(f"{OUTPUT_DIR}/test_no_lora.png")
print("✅ Saved: test_no_lora.png")
