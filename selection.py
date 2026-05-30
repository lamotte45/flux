from fastapi import APIRouter, Form

router = APIRouter()

@router.post("/confirm_style")
async def confirm_style(
    selected_image: str = Form(...),
    email: str = Form(...)
):
    return {
        "type": "3d_ready",
        "mesh_url": f"https://aibeautyconcepts.com/generated/{email.replace('@','_').replace('.','_')}.obj"
    }
