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

print("🔥 Loading SDXL...")

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL,
    torch_dtype=DTYPE
).to(DEVICE)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# --------------------------
# LOAD KOHYA LORA (MANUAL)
# --------------------------
print("🔥 Loading LoRA (kohya mode)...")

state_dict = load_file(LORA_PATH)

pipe.unet.load_attn_procs(state_dict)

print("✅ LoRA loaded")

# --------------------------
# PROMPTS
# --------------------------
tests = [
    ("basketball", "basketball seams shaved into fade"),
    ("lettering", "text design shaved into hair, word blessed"),
    ("geometric", "geometric pattern shaved into fade"),
]

NEGATIVE = "blurry, bad anatomy, tattoo"

print("🎨 Generating...")

for name, prompt in tests:

    full_prompt = f"""
    real barber haircut, african american man, side profile,
    clean low skin fade haircut,
    ultra sharp razor carved design in the fade,
    {prompt},
    realistic lighting, DSLR photo
    """

    print(f"➡️ {name}")

    image = pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=25,
        guidance_scale=7.5,
        width=1024,
        height=1024
    ).images[0]

    path = f"{OUTPUT_DIR}/{name}.png"
    image.save(path)

    print(f"✅ Saved: {path}")

print("🔥 DONE")
