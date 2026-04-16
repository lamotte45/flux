import torch
import os
from diffusers import FluxPipeline

# This looks for a variable on your system instead of hardcoding the secret
MY_TOKEN = os.getenv("HF_TOKEN")

if not MY_TOKEN:
    print("❌ Error: HF_TOKEN not found. Run 'export HF_TOKEN=your_token' first.")
    exit()

print("🚀 Starting Optimized FLUX on NVIDIA L4...")

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    token=MY_TOKEN
)

pipe.enable_model_cpu_offload()

prompt = "ultra realistic barber haircut, skin fade, razor design, DSLR lighting"

print("🎨 Generating...")
image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256
).images[0]

image.save("barber_output.png")
print("✅ SUCCESS!")
