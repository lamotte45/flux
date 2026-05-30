import torch
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os

# -----------------------------
# PATHS
# -----------------------------
OUTPUT_DIR = "training_data/generated_razor"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("🔥 Device:", DEVICE)

BASE_HEAD = "training_data/micro_geometric/10_designs/master_0056.png"

MICRO_LORA = "lora_outputs/micro_geometric/micro_geo_lora.safetensors"
RAZOR_LORA = "lora_outputs/razor_designs/razor_designs_lora-step00003000.safetensors"

# -----------------------------
# LOAD CONTROLNET + SDXL
# -----------------------------
print("🔥 Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
).to(DEVICE)

print("🔥 Loading SDXL Img2Img + ControlNet...")
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(DEVICE)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# -----------------------------
# LOAD LORAs
# -----------------------------
print("🔥 Loading LoRAs...")
pipe.load_lora_weights(MICRO_LORA, adapter_name="micro")
pipe.load_lora_weights(RAZOR_LORA, adapter_name="razor")

pipe.set_adapters(
    ["micro", "razor"],
    adapter_weights=[1.2, 1.6]
)

print("✅ Pipeline + LoRAs ready")

# -----------------------------
# BASE IMAGE + CONTROL IMAGE
# -----------------------------
base = Image.open(BASE_HEAD).convert("RGB").resize((1024, 1024))

# Canny from base to lock silhouette + fade
base_np = np.array(base)
edges = cv2.Canny(base_np, 80, 160)
control_img = Image.fromarray(edges).convert("RGB")

# -----------------------------
# PROMPTS
# -----------------------------
BASE_PROMPT = (
    "ultra realistic barber haircut photo, back of head, african american male, "
    "clean low skin fade, real human scalp texture, visible pores, slight imperfections, "
    "razor design shaved into the hair, exposed scalp inside the carved lines, "
    "high contrast between dark hair and shaved scalp, "
    "precise clipper work, crisp sharp lineup, "
    "DSLR photography, 85mm lens, shallow depth of field, "
    "plain neutral background"
)

NEGATIVE = (
    "cartoon, 3d render, CGI, fake skin, plastic skin, smooth plastic, "
    "illustration, painting, digital art, anime, stylized, "
    "tattoo, overlay, graphic design, logo floating on hair, text overlay, "
    "jewelry, earrings, chains, busy background, barbershop interior, "
    "blurry, low quality, noise, distortion, warped head, alien, creature"
)

DESIGNS = [
    "single curved wave razor pattern",
    "double parallel razor lines across the fade",
    "sharp zig zag carved pattern",
    "basketball seam style lines shaved into the fade",
    "triangle geometric razor fade design",
    "three parallel stripe razor pattern",
    "lightning bolt carved into the fade",
    "circular swirl razor pattern on the back",
    "minimal star shaved into the fade",
    "sharp straight taper razor lines"
]

# -----------------------------
# GENERATION LOOP (10 IMAGES)
# -----------------------------
for i, d in enumerate(DESIGNS[:10]):
    prompt = BASE_PROMPT + ", " + d
    print(f"🎨 Generating {i+1}/10 → {d}")

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        image=base,
        control_image=control_img,
        controlnet_conditioning_scale=0.72,
        strength=0.62,
        num_inference_steps=32,
        guidance_scale=7.8
    ).images[0]

    # Light enhancement only
    image = ImageEnhance.Contrast(image).enhance(1.12)
    image = ImageEnhance.Sharpness(image).enhance(1.18)

    path = f"{OUTPUT_DIR}/razor_{i:03d}.png"
    image.save(path)
    print(f"✅ Saved: {path}")

print("🔥 DONE — 10 controlled razor designs generated")
