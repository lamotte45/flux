from fastapi import APIRouter
from pydantic import BaseModel
from process_hair import run_hair_analysis

router = APIRouter(prefix="/avatar", tags=["avatar"])

class AvatarRequest(BaseModel):
    user_text: str
    session_id: str | None = None

class AvatarResponse(BaseModel):
    avatar_reply: str
    session_id: str

def generate_avatar_reply(user_text: str) -> str:
    text = user_text.lower()

    # --- Detect hair description keywords ---
    hair_keywords = ["short", "long", "wavy", "curly", "black", "brown", "blonde", "streak"]

    if any(word in text for word in hair_keywords):
        analysis = run_hair_analysis(user_text)
        streak_text = ", ".join(analysis['streaks']) if analysis['streaks'] else "none"
        return (
            f"I've analyzed your hair. "
            f"Length: {analysis['hair_length']}, "
            f"Texture: {analysis['hair_texture']}, "
            f"Base color: {analysis['base_color']}, "
            f"Streaks: {streak_text}. "
            "Shall I prepare six style variations for you?"
        )

    return "I understand. Give me a moment while I prepare six hairstyle options."

@router.post("/talk", response_model=AvatarResponse)
async def avatar_talk(payload: AvatarRequest):
    reply = generate_avatar_reply(payload.user_text)
    session = payload.session_id or "session_" + str(abs(hash(payload.user_text)) % 10_000_000)
    return AvatarResponse(avatar_reply=reply, session_id=session)
