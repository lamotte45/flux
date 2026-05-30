import torch
import os
from diffusers import StableDiffusionPipeline

os.makedirs("outputs", exist_ok=True)

print("⏳ Loading Pipeline (Fidelity Fix V4)...")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

print("⏳ Applying LoRA at 0.7 strength (Optimized for Texture + Skin)...")
# Using the adapter weight method to find the perfect mix of base model shape and trained texture
pipe.load_lora_weights("models/hair_beauty_final.safetensors", adapter_name="hair")
pipe.set_adapters(["hair"], adapter_weights=[0.70])

# Detailed prompt focusing on AI Beauty Concepts texture standards and framing
# Crucially, we use "voluminous" and "separate strands" with a correct aspect ratio
prompt = (
    "a professional high-resolution medium-shot studio photo of hairbeauty, "
    "kinky straight hair texture, full voluminous natural black hair, "
    "crisply separate hair strands, highly detailed pores, natural realistic skin, 8k, "
    "soft studio lighting, natural reflections"
)

negative_prompt = (
    "etched hair, drawn on skin, compressed hair, chopped off hair, flat, "
    "helmet, plastic, shiny, anime, cartoon skin, over-smoothed skin, blurry, distorteds"
)

print("🎨 Rendering with high-fidelity skin and hair details...")
with torch.inference_mode():
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=60, # Bumped steps to resolve skin and hair detail
        guidance_scale=6.5, # Reduced slightly to prevent over-smoothing
        height=768, # Increased height to prevent "chopped" hair
        width=512    # Traditional portrait aspect ratio
    ).images[0]

# Saved with a new name to compare
image.save("outputs/pro_render_fidelity.png")
print("✅ SUCCESS: Image saved to outputs/pro_render_fidelity.png")
