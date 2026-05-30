import torch
from diffusers import StableDiffusionXLPipeline
import os
from datetime import datetime

# -------------------------------
# CONFIG
# -------------------------------
DEVICE = "cuda"

MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH = "/home/kenny/barber_ai/models/lora/barber_design_v2-step00001500.safetensors"

OUTPUT_DIR = "/home/kenny/barber_ai/generated_styles"
LOG_FILE = "/home/kenny/barber_ai/generation_log.txt"

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

print("✅ System Ready")

# -------------------------------
# NEGATIVE PROMPT
# -------------------------------
NEGATIVE = (
    "cartoon, anime, cgi, 3d render, illustration, vector, "
    "blurry, low quality, distorted, bad anatomy, "
    "plastic skin, waxy skin, over-smoothed skin, "
    "messy fade, uneven fade, patchy hair, "
    "blurry hair design, melted lines, distorted pattern, "
    "tattoos, jewelry, accessories"
)

# -------------------------------
# DESIGN SYSTEM
# -------------------------------
def build_design(design_type):
    if design_type == "basketball":
        return (
            "basketball seams shaved into fade, "
            "curved intersecting razor lines forming basketball pattern, "
            "pebbled texture effect inside shaved sections"
        )

    if design_type == "soccer":
        return (
            "hexagon and pentagon pattern shaved into fade, "
            "clean geometric razor lines forming soccer ball panels"
        )

    if design_type == "flame":
        return (
            "flowing flame shapes shaved into fade, "
            "curved tapering razor lines with motion effect"
        )

    if design_type == "lightning":
        return (
            "sharp zig-zag lightning bolt shaved into fade, "
            "angular razor lines with high contrast edges"
        )

    return "clean razor line design shaved into fade"

# -------------------------------
# PROMPT BUILDER
# -------------------------------
def build_prompt(design_type):

    design = build_design(design_type)

    prompt = (
        "real barbershop photo, african american male, side profile, head and shoulders, "
        "tight coarse curls, dense coil texture, natural hair pattern, "
        "low taper fade, sharp lineup, clean edge up, "

        f"{design}, "
        "high contrast shaved design, visible scalp in shaved areas, "

        "precise barber hair art, realistic hair density transition, "
        "visible skin pores, natural imperfections, "

        "authentic barbershop environment, handheld camera feel, "
        "iPhone 15 photo quality, natural lighting, shallow depth of field, "
        "ultra realistic, high detail"
    )

    return prompt

# -------------------------------
# GENERATION LOOP
# -------------------------------
DESIGNS = ["basketball", "soccer", "flame", "lightning"]

count = 0

print("🎨 Generating styles...")

for design_type in DESIGNS:
    for i in range(5):

        prompt = build_prompt(design_type)

        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=25,
            guidance_scale=7.0,
            width=1024,
            height=1024
        ).images[0]

        filename = f"pro_style_{count:04d}.png"
        filepath = f"{OUTPUT_DIR}/{filename}"

        image.save(filepath)

        # Save prompt log
        with open(LOG_FILE, "a") as f:
            f.write(f"{datetime.now()} | {filename} | {design_type}\n{prompt}\n\n")

        print(f"✅ Generated: {filename}")

        count += 1

print("🔥 DONE — ALL STYLES GENERATED")
