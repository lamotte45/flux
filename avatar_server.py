from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import base64
import torch
from diffusers import AutoPipelineForImage2Image

app = FastAPI()

# 🔥 LOAD MODEL ON START
pipe = AutoPipelineForImage2Image.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

print("🔥 SDXL Image2Image READY")

@app.post("/avatar/request")
async def avatar_request(
    image: UploadFile = File(...),
    text: str = Form(...)
):
    try:
        # ✅ LOAD IMAGE
        contents = await image.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_image = input_image.resize((1024, 1024))

        print("🔥 IMAGE RECEIVED:", input_image.size)
        print("🔥 PROMPT:", text)

        # ✅ FORCE IMAGE USAGE (THIS FIXES THE CHAIR ISSUE)
        result = pipe(
            prompt=f"{text}, same person, same face, realistic human portrait, do not change identity",
            image=input_image,
            strength=0.5,
            guidance_scale=7.5
        ).images[0]

        # ✅ CONVERT TO BASE64
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return {"result_image": img_str}

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}

