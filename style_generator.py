import os
import uuid
import torch
from fastapi import APIRouter
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

router = APIRouter(prefix="/styles", tags=["styles"])

# --- RTX 4090 HIGH-DEFINITION LOADING ---
try:
    # We switch from 'Turbo' to 'Base 1.0' for perfect eyes and faces
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    
    # Use a faster scheduler for the 4090
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    GPU_READY = True
    print("RTX 4090: HD Engine Online (Perfect Faces Mode)")
except Exception as e:
    print(f"GPU Error: {e}")
    GPU_READY = False

class StyleRequest(BaseModel):
    prompt: str
    session_id: str | None = None
    hair_length: str = "medium"
    hair_texture: str = "straight"
    base_color: str = "natural"
    streaks: str = "none"

class StyleResponse(BaseModel):
    session_id: str
    image_urls: list[str]
    refined_prompt: str

def build_luxury_prompt(req: StyleRequest) -> str:
    # We add 'symmetrical eyes' and 'detailed pupils' to the master prompt
    return (
        f"Professional luxury salon photography, a woman with {req.hair_length} "
        f"{req.hair_texture} {req.base_color} hair, featuring elegant {req.streaks}, "
        f"highly detailed face, symmetrical eyes, focused gaze, 8k resolution, "
        f"cinematic lighting, {req.prompt}"
    )

@router.post("/generate", response_model=StyleResponse)
async def generate_styles(payload: StyleRequest):
    session = payload.session_id or str(uuid.uuid4())[:8]
    master_prompt = build_luxury_prompt(payload)
    negative_prompt = "crossed eyes, deformed iris, blurry eyes, bad anatomy, distorted face"
    
    base_dir = f"/var/www/html/generated/{session}"
    os.makedirs(base_dir, exist_ok=True)
    urls = []
    
    if GPU_READY:
        for i in range(6):
            # 25 steps on a 4090 takes ~1 second per image. 
            # This fixes the 'crossed eyes' issue.
            image = pipe(
                prompt=master_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25, 
                guidance_scale=7.5
            ).images[0]
            
            filename = f"style_{i}.png"
            filepath = os.path.join(base_dir, filename)
            image.save(filepath)
            urls.append(f"/generated/{session}/{filename}")
    
    return StyleResponse(
        session_id=session, 
        image_urls=urls,
        refined_prompt=master_prompt
    )
