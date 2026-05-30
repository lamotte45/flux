from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional

from hair_analysis import run_hair_analysis
from style_generator import generate_6_styles
from appointment_scheduler import schedule_appointment
from emotion_engine import run_emotion_engine
from detect_intent import detect_intent

router = APIRouter()

@router.post("/concierge")
async def concierge_router(
    image: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    intent = detect_intent(text)

    if intent == "emotion":
        result = run_emotion_engine(image, audio, text)
        return {
            "module": "emotion_engine",
            "emotion_score": result["emotion_score"],
            "best_style": result["best_style"]
        }

    if intent == "hair_analysis":
        result = run_hair_analysis(image)
        return {
            "module": "hair_analysis",
            "recommendations": result
        }

    if intent == "style_generation":
        images = generate_6_styles(image, text)
        return {
            "module": "style_generator",
            "images": images
        }

    if intent == "appointment":
        confirmation = schedule_appointment(text)
        return {
            "module": "appointment_scheduler",
            "confirmation": confirmation
        }

    return {
        "module": "unknown",
        "message": "I didn't understand your request."
    }
