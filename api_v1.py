import torch
import gc
from fastapi import FastAPI, File, UploadFile, Form
from diffusers import AutoPipelineForInpainting
from PIL import Image, UnidentifiedImageError
import io
import base64
import numpy as np

app = FastAPI()

# 🔥 LOAD REAL INPAINTING MODEL (NOT TURBO)
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

print("🔥 SDXL INPAINTING READY")

@app.post("/generate_styles")
async def generate(image: UploadFile = File(...), prompt: str = Form(...)):
    try:
        gc.collect()
        torch.cuda.empty_cache()

        content = await image.read()
        if not content or len(content) < 100:
            return {"error": "Invalid image"}

        try:
            input_image = Image.open(io.BytesIO(content)).convert("RGB")
        except UnidentifiedImageError:
            return {"error": "Bad image format"}

        input_image = input_image.resize((1024, 1024))

        # 🔥 SIMPLE FULL MASK (FOR NOW)
        mask = Image.fromarray(
            np.ones((1024, 1024), dtype=np.uint8) * 255
        )

        print("🔥 RUNNING INPAINTING")

        result = pipe(
            prompt=f"{prompt}, platinum blonde finger waves hairstyle, same person, same face, realistic barber haircut",
            image=input_image,
            mask_image=mask,
            strength=0.95,
            guidance_scale=7.5,
            num_inference_steps=30
        ).images[0]

        buffer = io.BytesIO()
        result.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return {"result_image": img_str}

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
