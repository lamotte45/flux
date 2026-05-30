import os
import uuid
import torch
from diffusers import AutoPipelineForText2Image

OUTPUT_DIR = "/var/www/html/generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔥 GENERATOR READY (CLEAN MODE)")

# ================================
# BASE MODEL (SDXL BASE)
# ================================
base_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# ================================
# REAL MODEL (LOCAL ONLY)
# ================================
REAL_MODEL_PATH = "/home/user/barber_ai/real_model"

try:
    real_pipe = AutoPipelineForText2Image.from_pretrained(
        REAL_MODEL_PATH,
        torch_dtype=torch.float16
    ).to("cuda")
    REAL_MODEL_AVAILABLE = True
    print("🔥 REAL MODEL LOADED")
except Exception:
    REAL_MODEL_AVAILABLE = False
    print("⚠️ REAL MODEL NOT FOUND — USING BASE MODEL INSTEAD")

def _save_and_url(img):
    file_id = str(uuid.uuid4()) + ".jpg"
    path = os.path.join(OUTPUT_DIR, file_id)
    img.save(path)
    return f"https://aibeautyconcepts.com/generated/{file_id}"

# ================================
# BASE MODEL GENERATION
# ================================
def generate_model_styles(text: str):
    images = []
    for _ in range(4):
        img = base_pipe(
            prompt=f"ultra realistic portrait, {text}, African American hair texture, studio lighting",
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        images.append(_save_and_url(img))
    return images

# ================================
# REAL MODEL GENERATION
# ================================
def generate_real_model(text: str):
    if not REAL_MODEL_AVAILABLE:
        return generate_model_styles(text)

    images = []
    for _ in range(4):
        img = real_pipe(
            prompt=f"{text}, ultra realistic, African American textures, premium detail",
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        images.append(_save_and_url(img))
    return images
