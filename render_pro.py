import torch
from diffusers import StableDiffusionPipeline
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

model_id = "runwayml/stable-diffusion-v1-5"
lora_path = "models/hair_beauty_final.safetensors"

print("⏳ Loading Pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

print(f"⏳ Loading LoRA at 0.65 strength (Preventing Helmet Effect)...")
pipe.load_lora_weights(lora_path, adapter_name="hair")
# We set the weight lower to let the hair "breathe" away from the scalp
pipe.set_adapters(["hair"], adapter_weights=[0.65])

# Detailed prompt focusing on AI Beauty Concepts texture standards
prompt = (
    "a high-resolution studio photo of hairbeauty, kinky straight hair texture, "
    "voluminous natural black hair, soft realistic hairline, separate hair strands, "
    "highly detailed, professional lighting, 8k"
)

negative_prompt = (
    "etched hair, tattoo hair, drawn on skin, flat hair, plastic, "
    "helmet, shiny, low resolution, blurry, distorted scalp"
)

print("🎨 Rendering high-fidelity texture...")
with torch.inference_mode():
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=50, # More steps for finer 4C/Kinky detail
        guidance_scale=7.0
    ).images[0]

image.save("outputs/pro_render_065.png")
print("✅ SUCCESS: Image saved to outputs/pro_render_065.png")
