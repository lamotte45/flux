import torch
from diffusers import StableDiffusionPipeline
import os

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

model_id = "runwayml/stable-diffusion-v1-5"
lora_path = "models/hair_beauty_final.safetensors"

print("⏳ Loading Pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

print("⏳ Applying LoRA (Weight 0.7)...")
pipe.load_lora_weights(lora_path, adapter_name="hair")
pipe.set_adapters(["hair"], adapter_weights=[0.7])

# The prompt now includes "centered" and "empty space above head" 
# to force the AI to move the subject down.
prompt = (
    "a professional medium shot photo of hairbeauty, "
    "full head visible, empty space above the head, "
    "voluminous kinky straight hair, natural black color, "
    "realistic skin, studio background, high quality, 8k"
)

# Added "out of frame" and "cropped" to negative prompt
negative_prompt = (
    "cropped head, chopped off hair, top of head missing, "
    "close up, zoom, face shot, etched hair, plastic, cartoon"
)

print("🎨 Rendering with full vertical framing...")
with torch.inference_mode():
    # Height 896 + Width 512 creates a 9:16 portrait style
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=50, 
        guidance_scale=7.5,
        height=896, 
        width=512
    ).images[0]

image.save("outputs/full_head_capture.png")
print("✅ SUCCESS: Image saved to outputs/full_head_capture.png")
