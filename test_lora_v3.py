import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("🔥 DEVICE:", DEVICE)

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH = "/home/kenny/barber_ai/lora_outputs/razor_designs/razor_designs_lora.safetensors"

OUTPUT_DIR = "/home/kenny/barber_ai/test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL,
    torch_dtype=DTYPE
).to(DEVICE)

pipe.enable_attention_slicing()

state_dict = load_file(LORA_PATH)
pipe.unet.load_attn_procs(state_dict)

print("✅ LoRA loaded")

NEGATIVE = """
beard design, cheek patch, face markings,
tattoo, logo, overlay,
photo realistic perfection,
blurry, messy fade
"""

def build_prompt(style):
    return f"""
    barber haircut, side profile,
    VERY OBVIOUS LARGE GEOMETRIC DESIGN IN HAIR,
    shaved pattern in fade,
    grid pattern CLEARLY VISIBLE,
    bold carved lines,
    thick razor cuts,
    high contrast shaved vs hair,
    simple background,
    focus on haircut design,
    {style}
    """

tests = [
    ("grid",      "square grid pattern"),
    ("zigzag",    "zig zag pattern"),
    ("triangles", "triangle pattern")
]

print("🎨 FORCING STRUCTURE...")

for name, style in tests:
    prompt = build_prompt(style)
    print(f"➡️ {name}")
    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=35,
        guidance_scale=11.0,
        width=1024,
        height=1024
    ).images[0]
    path = f"{OUTPUT_DIR}/v3_{name}.png"
    image.save(path)
    print(f"✅ Saved: {path}")

print("🔥 DONE")
