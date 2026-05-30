import os
import torch
from diffusers import FluxPipeline

# -----------------------------
# OUTPUT DIRECTORY
# -----------------------------
OUTPUT_DIR = "./training_data/generated_flux"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("⚡ Loading Flux.1-dev (L4 Turbo Mode)...")

# -----------------------------
# LOAD FLUX (L4-OPTIMIZED)
# -----------------------------
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Memory / speed optimizations
pipe.enable_attention_slicing()
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("✅ Using xFormers memory-efficient attention")
except Exception:
    print("⚠️ xFormers not available, continuing without it")

print("✅ Flux.1-dev loaded on CUDA")

# -----------------------------
# PROMPTS
# -----------------------------
BASE_PROMPT = (
    "extreme close up barber razor design, shaved skin fade, 3D depth, "
    "realistic follicles, ultra realistic skin texture, 8k, studio lighting"
)

DESIGNS = [
    "sharp zigzag razor pattern",
    "lightning bolt razor design",
    "smooth wave razor pattern",
    "triangle geometric razor design",
    "minimal star carved into fade",
    "cross razor pattern in the fade",
    "circular swirl razor design",
    "three parallel razor lines",
    "tribal inspired razor pattern",
    "clean minimal single line design",
]

# -----------------------------
# GENERATION LOOP
# -----------------------------
print("🎨 Starting batch generation...")
with torch.inference_mode():
    for i, design in enumerate(DESIGNS):
        prompt = f"{BASE_PROMPT}, {design}"
        print(f"📸 Generating {i+1}/{len(DESIGNS)} → {design}...")

        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=2.8,      # Slightly lower for speed + realism
            num_inference_steps=20,  # Faster but still high quality
        ).images[0]

        filename = f"razor_{i:02d}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        image.save(save_path)

        print(f"✅ Saved: {save_path}")

print("🏁 Done. All images generated.")
