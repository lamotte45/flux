import torch
from diffusers import StableDiffusionXLPipeline
import os

# -------------------------------
# CONFIG
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH = "/home/kenny/barber_ai/models/lora/barber_design_v2-step00001500.safetensors"

OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
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

# -------------------------------
# PROMPT (YOUR NEW VERSION)
# -------------------------------
PROMPT = """
RAW photo, studio headshot, african american male, back of head view,

low skin fade haircut, smooth short hair on top, no curls, clean scalp texture,
sharp lineup, clean taper,

large circular shaved design centered on the back of the head,
round shape with curved intersecting arcs,
one vertical arc and one horizontal arc crossing through the center,
two outer curved rib lines wrapping around the circle,
symmetrical spherical geometry resembling a basketball,

deep razor carved lines, high contrast shaved design,
clean crisp edges, design clearly visible and centered,

realistic skin texture, visible pores, natural imperfections,
no tattoos, no ink, no jewelry,

plain neutral background, studio lighting,
shallow depth of field, ultra realistic, high detail
"""

NEGATIVE = """
tattoo, ink, faint design, blurry design, low contrast design,
cartoon, anime, illustration, vector, 3d render,
plastic skin, waxy skin, fake lighting, messy hair, curls
"""

# -------------------------------
# GENERATE
# -------------------------------
print("🎨 Generating...")

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    num_inference_steps=30,
    guidance_scale=7.0,
    width=1024,
    height=1024
).images[0]

# -------------------------------
# SAVE
# -------------------------------
output_path = f"{OUTPUT_DIR}/backview_basketball.png"
image.save(output_path)

print(f"✅ Saved: {output_path}")
