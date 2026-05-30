import os
from huggingface_hub import InferenceClient

# -----------------------------
# LOAD TOKEN FROM ENV
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN not set. Run: export HF_TOKEN=your_token")

# -----------------------------
# INIT CLIENT
# -----------------------------
client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN,
)

# -----------------------------
# PROMPT (BARBER TEST)
# -----------------------------
prompt = (
    "african american man, side profile, clean low fade haircut, "
    "sharp geometric hair design, razor lines, precise barber carving, "
    "high detail, realistic lighting"
)

print("🎨 Generating with Flux (fal-ai)...")

# -----------------------------
# GENERATE IMAGE
# -----------------------------
image = client.text_to_image(
    prompt,
    model="black-forest-labs/FLUX.1-dev",
)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
output_path = "/home/kenny/barber_ai/generated_styles/flux_fal.png"
image.save(output_path)

print(f"✅ Saved: {output_path}")
print("🔥 DONE")
