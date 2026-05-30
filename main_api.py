import torch
from fastapi import FastAPI, UploadFile, File, Form
from diffusers import AutoPipelineForImage2Image
from PIL import Image
import io, base64

app = FastAPI()

# =========================================
# 🔥 LOAD MODEL
# =========================================
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_attention_slicing()

print("🔥 MAIN AI SYSTEM READY")

# =========================================
# 🧠 STYLE DICTIONARY
# =========================================
STYLE_PRESETS = {
    "mid_fade": {
        "keywords": ["mid fade","fade haircut"],
        "prompt": "professional mid fade haircut, clean taper, sharp lineup"
    },
    "blonde": {
        "keywords": ["blonde","balayage"],
        "prompt": "luxury blonde balayage hair, sun kissed highlights"
    },
    "curls": {
        "keywords": ["curly","curls"],
        "prompt": "big bouncy salon curls, high volume, glossy finish"
    }
}

# =========================================
# 🧠 INTENT DETECTION
# =========================================
def detect_intent(text):
    text = text.lower()

    if "book" in text:
        return "BOOK_APPOINTMENT"

    if "hair" in text or "fade" in text or "style" in text:
        return "STYLE_REQUEST"

    return "GENERAL_CHAT"

# =========================================
# 🧠 STYLE INTERPRETER
# =========================================
def interpret_style(text):
    for key, value in STYLE_PRESETS.items():
        for keyword in value["keywords"]:
            if keyword in text:
                return value["prompt"]

    return "clean modern professional haircut"

# =========================================
# 🎨 GENERATE 6 STYLES
# =========================================
def generate_styles(image, prompt):
    results = []

    img = image.resize((768,768))

    for i in range(6):
        output = pipe(
            prompt=f"ultra realistic portrait, {prompt}, detailed hair, same face",
            image=img,
            strength=0.7,
            guidance_scale=0.0,
            num_inference_steps=2
        ).images[0]

        buf = io.BytesIO()
        output.save(buf, format="JPEG")

        results.append(base64.b64encode(buf.getvalue()).decode())

    return results

# =========================================
# 🚀 /generate_styles
# =========================================
@app.post("/generate_styles")
async def generate_styles_api(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    content = await image.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    images = generate_styles(img, prompt)

    return {"images": images}

# =========================================
# 🧠 /brain (MAIN ENTRY POINT)
# =========================================
@app.post("/brain")
async def brain(
    image: UploadFile = File(...),
    text: str = Form(...)
):
    intent = detect_intent(text)

    if intent == "STYLE_REQUEST":
        prompt = interpret_style(text)

        content = await image.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        images = generate_styles(img, prompt)

        return {
            "type": "style_preview",
            "images": images
        }

    if intent == "BOOK_APPOINTMENT":
        return {
            "type": "booking",
            "message": "Booking feature coming soon"
        }

    return {
        "type": "chat",
        "message": "How can I help you today?"
    }

# =========================================
# 🚀 START SERVER
# =========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
