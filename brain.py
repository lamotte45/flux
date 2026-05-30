from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io

from generator import generate_styles
from styles import detect_intent, interpret_request

router = APIRouter()

@router.post("/avatar_brain")
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
        return {"type": "booking"}

    return {"type": "chat", "message": "How can I help you?"}
