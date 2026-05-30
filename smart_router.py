from fastapi import APIRouter
from pydantic import BaseModel
from smart_suggestions import smart_suggest

router = APIRouter(prefix="/smart", tags=["smart"])

class SuggestRequest(BaseModel):
    style_text: str

@router.post("/suggest")
async def suggest(payload: SuggestRequest):
    return smart_suggest(payload.style_text)
