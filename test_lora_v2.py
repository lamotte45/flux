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
# LOAD LORA (KOHYA SAFE)
# --------------------------
print("🔥 Loading LoRA...")
state_dict = load_file(LORA_PATH)
pipe.unet.load_attn_procs(state_dict)

print("✅ LoRA loaded")

# --------------------------
# NEGATIVE PROMPT (STRONG)
# --------------------------
NEGATIVE = """
beard design, beard patch, cheek design, face markings,
tattoo, painted design, graphic overlay,
blurry, distorted, bad anatomy, messy fade
"""

# --------------------------
# FORCE DESIGN PROMPT
# --------------------------
def build_prompt(style):

    return f"""
    ((real barber haircut)), african american man, side profile,
    
    ((clean low skin fade haircut)),

    (((ULTRA SHARP GEOMETRIC DESIGN SHAVED INTO THE FADE ABOVE THE EAR))),
    (((DESIGN CARVED INTO HAIR USING CLIPPERS))),
    (((HIGH CONTRAST BETWEEN SHAVED AND UNSHAVED HAIR))),

    (({style})),

    no beard design,
    no face design,
    no cheek patterns,
    design ONLY inside fade region,

    realistic hair texture, natural lighting, DSLR photo, high detail
    """

# --------------------------
# TEST STYLES
# --------------------------
tests = [
    ("geometric", "triangles and sharp angular shapes"),
    ("zigzag", "zig-zag lightning pattern"),
    ("grid", "grid pattern carved into fade"),
]

# --------------------------
# GENERATION LOOP
# --------------------------
print("🎨 Generating FORCED designs...")

for name, style in tests:

    prompt = build_prompt(style)

    print(f"➡️ {name}")

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=30,
        guidance_scale=9.0,   # 🔥 HIGHER GUIDANCE
        width=1024,
        height=1024
    ).images[0]

    path = f"{OUTPUT_DIR}/forced_{name}.png"
    image.save(path)

    print(f"✅ Saved: {path}")

print("🔥 DONE")
