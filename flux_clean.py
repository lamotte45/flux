import torch
from diffusers import FluxPipeline
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🔥 Loading Flux...")

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
).to(DEVICE)

pipe.enable_attention_slicing()

print("✅ Flux ready")

# -----------------------------
# PROMPT (BARBER TEST)
# -----------------------------
prompt = (
    "african american man, side profile, clean low fade haircut, "
    "sharp geometric hair design, precise barber lines, "
    "realistic hair texture, high detail, studio lighting"
)

print("🎨 Generating...")

image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    num_inference_steps=28
).images[0]

output_path = "/home/kenny/barber_ai/generated_styles/flux_clean.png"
image.save(output_path)

print(f"✅ Saved: {output_path}")
print("🔥 DONE")
