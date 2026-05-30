import torch
from diffusers import StableDiffusionXLPipeline
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH = "/home/kenny/barber_ai/models/lora/barber_design_v2-step00001500.safetensors"

OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
pipe.fuse_lora(lora_scale=1.0)

print("✅ Ready")

PROMPT = """
real barbershop photo, african american boy, backhead profile,
low skin fade haircut, short textured curls on top, sharp lineup,
high-contrast basketball design shaved deeply into the fade,
bold razor lines, clean crisp edges, design centered and clearly visible,
realistic skin texture, visible pores, natural imperfections,
no tattoos, no ink on skin, no jewelry,
authentic barbershop background, natural lighting,
handheld camera feel, iPhone 15 photo quality,
shallow depth of field, <lora:barber_design_v2:1.4>
"""

NEGATIVE = """
tattoo, ink, faint design, blurry design, low contrast design,
cartoon, anime, illustration, vector, 3d render, plastic skin,
waxy skin, fake lighting, smooth skin, studio render
"""

print("🎨 Generating...")

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    num_inference_steps=30,
    guidance_scale=7.0,
    width=1024,
    height=1024
).images[0]

output_path = f"{OUTPUT_DIR}/exact_prompt.png"
image.save(output_path)

print(f"✅ Saved: {output_path}")
