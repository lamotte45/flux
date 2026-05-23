import boto3
import base64
import json
import os
from PIL import Image, ImageOps, ImageFilter
import io
from datetime import date

# === OUTPUT DIR ===
OUTPUT_DIR = f"/home/kenny/barber_ai/outputs/bedrock_{date.today().strftime('%Y%m%d')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = boto3.client("bedrock-runtime", region_name="us-west-2")

# === PATHS ===
FADE_PATH = "/home/kenny/barber_ai/assets/fade_base.png"
KOBE_PATH = "/home/kenny/barber_ai/kobe_ref.jpg"

# === LOAD FADE ===
fade = Image.open(FADE_PATH).convert("RGB").resize((1024, 1024))

# === LOAD & PREP KOBE STENCIL ===
kobe = Image.open(KOBE_PATH).convert("L").resize((360, 360))
kobe = kobe.filter(ImageFilter.FIND_EDGES)
kobe = ImageOps.invert(kobe)

# Mask: only dark lines transfer
mask = kobe.point(lambda x: 255 if x < 200 else 0)

# === COMPOSITE ===
PASTE_X, PASTE_Y = 400, 160
fade.paste(kobe, (PASTE_X, PASTE_Y), mask)

# Encode composite for Bedrock img2img
buffer = io.BytesIO()
fade.save(buffer, format="PNG")
encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

# === PROMPTS ===
PROMPT = (
    "RazorArtStyle, ultra realistic barber hair engraving, "
    "portrait carved into skin fade using hair stubble shading, "
    "sharp razor lines, photorealistic, 8k detail"
)
NEG = "tattoo, ink, sticker, cartoon, blurry, plastic skin, logo, text"

# === CALL BEDROCK IMG2IMG ===
response = client.invoke_model(
    modelId="stability.sd3-5-large-v1:0",
    body=json.dumps({
        "prompt": PROMPT,
        "negative_prompt": NEG,
        "mode": "image-to-image",
        "image": encoded_image,
        "strength": 0.32,
        "output_format": "png"
    })
)

result = json.loads(response["body"].read())
final_img = Image.open(io.BytesIO(base64.b64decode(result["images"][0])))

OUTFILE = f"{OUTPUT_DIR}/kobe_engraved.png"
final_img.save(OUTFILE)

print(f"ENGRAVING COMPLETE → {OUTFILE}")
