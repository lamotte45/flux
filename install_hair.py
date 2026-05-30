import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from pathlib import Path
import os

# ---- CONFIG ----
# Specialized inpainting model to prevent artifacts
BASE_MODEL = "runwayml/stable-diffusion-inpainting"
LORA_PATH = "/home/kenny/barber_ai/models/lora/ABC_KinkyStraight_v2-000002.safetensors"

IMG_NAME = "test1"
IMAGE_PATH = f"/home/kenny/barber_ai/training_data/lora_input/20_abcstyle/{IMG_NAME}.png"
MASK_PATH = f"/home/kenny/barber_ai/masks/head_sam/{IMG_NAME}_mask.png"

OUT_DIR = Path("/home/kenny/barber_ai/outputs/inpainting_tests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Focused purely on the HAIR texture
PROMPT = (
    "photorealistic studio portrait, abcstyle hair, long kinky straight hair, "
    "voluminous rounded top, highly detailed hair strands, 8k, professional lighting"
)

# Aggressive negatives to stop 'extra faces' and jewelry appearing in the hair
NEGATIVE_PROMPT = (
    "extra face, extra eyes, teeth, jewelry, lips, mouth, skin, "
    "distorted, cartoon, flat top, boxy hair, horizontal line, blurry, lowres"
)

device = "cuda"

# ---- PIPELINE ----
print("🕒 Loading specialized Inpainting Pipeline...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

print(f"🧬 Loading LoRA: {LORA_PATH}")
pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=0.80)

# ---- PREP ----
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: {IMAGE_PATH} not found")
    exit()

init_image = Image.open(IMAGE_PATH).convert("RGB").resize((512, 512))
mask_image = Image.open(MASK_PATH).convert("L").resize((512, 512))

# ---- GENERATION ----
print(f"🎨 Clean Volume Install on {IMG_NAME}...")
generator = torch.Generator(device=device).manual_seed(1234)

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=50,
    strength=0.85,          # High enough to add volume, low enough to respect the head shape
    guidance_scale=12.0,    # Strong guidance to force the AI to avoid the negative prompt items
    generator=generator,
).images[0]

out_path = OUT_DIR / f"{IMG_NAME}_ABC_CleanVolume.png"
image.save(out_path)

print(f"\n✅ Success! Clean result saved to: {out_path}")
