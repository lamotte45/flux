from fastapi import APIRouter, Form
from generator import generate_model_styles, generate_real_model

router = APIRouter()

@router.post("/concierge")
async def concierge(
    text: str = Form(...),
    email: str = Form(...),
    use_model: str = Form("base")
):
    print("🔥 MODE:", use_model)

    if use_model == "real":
        images = generate_real_model(text)
        return {
            "type": "model_preview",
            "mode": "real",
            "images": images
        }

    images = generate_model_styles(text)
    return {
        "type": "model_preview",
        "mode": "base",
        "images": images
    }
