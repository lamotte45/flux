import torch
from fastapi import FastAPI, UploadFile, File, Form
from diffusers import AutoPipelineForImage2Image
from PIL import Image
import io, base64, uuid, os

app = FastAPI()

OUTPUT_DIR = "/var/www/html/generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# 🔥 LOAD MODEL
# =========================================
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_attention_slicing()

print("🔥 AVATAR CORE ONLINE")

# =========================================
# 🧠 STYLE DICTIONARY
# =========================================
STYLE_PRESETS = {
    "mid_fade": {
        "keywords": ["mid fade","fade haircut"],
        "prompt": "professional mid fade haircut, clean taper, sharp lineup, studio lighting"
    },
    "balayage": {
        "keywords": ["balayage","blonde"],
        "prompt": "luxury blonde balayage, soft highlights, salon quality hair"
    },
    "curls": {
        "keywords": ["curly","curls"],
        "prompt": "big bouncy curls, high volume, glossy natural texture"
    }
}

# =========================================
# 🧠 INTENT
# =========================================
def detect_intent(text):
    text = text.lower()

    if "book" in text:
        return "BOOK"

    if "hair" in text or "fade" in text or "style" in text:
        return "STYLE"

    return "CHAT"

# =========================================
# 🧠 STYLE INTERPRETATION
# =========================================
def interpret_request(text):
    for key, val in STYLE_PRESETS.items():
        for word in val["keywords"]:
            if word in text:
                return val["prompt"]

    return "clean modern professional haircut"

# =========================================
# 🎨 GENERATION ENGINE
# =========================================
def generate_styles(image, prompt):
    img = image.resize((768,768))
    urls = []

    for i in range(6):
        result = pipe(
            prompt=f"ultra realistic portrait, {prompt}, same person, preserve identity",
            image=img,
            strength=0.7,
            guidance_scale=0.0,
            num_inference_steps=2
        ).images[0]

        file_id = str(uuid.uuid4()) + ".jpg"
        path = os.path.join(OUTPUT_DIR, file_id)

        result.save(path)

        urls.append(f"https://aibeautyconcepts.com/generated/{file_id}")

    return urls

# =========================================
# 🧠 AVATAR BRAIN (REAL CORE)
# =========================================
@app.post("/avatar_brain")
async def avatar_brain(
    image: UploadFile = File(...),
    text: str = Form(...)
):
    intent = detect_intent(text)

    if intent == "STYLE":
        prompt = interpret_request(text)

        content = await image.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        images = generate_styles(img, prompt)

        return {
            "type": "style_preview",
            "images": images
        }

    if intent == "BOOK":
        return {
            "type": "booking",
            "message": "Booking system coming next"
        }

    return {
        "type": "chat",
        "message": "How can I help you today?"
    }

# =========================================
# 🚀 START
# =========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
