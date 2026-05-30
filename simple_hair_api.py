import torch
from fastapi import FastAPI, UploadFile, File, Form, Request
from diffusers import AutoPipelineForInpainting
from PIL import Image
import io, base64, datetime
import numpy as np

# 🔥 IMPORT (as requested)
from email_service import send_admin_notification, send_user_receipt

app = FastAPI()

# 🔥 LOAD MODEL
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_attention_slicing()

# =========================================
# 💇 HAIR ENGINE
# =========================================
@app.post("/hair")
async def hair(image: UploadFile = File(...), prompt: str = Form(...)):
    try:
        content = await image.read()
        img = Image.open(io.BytesIO(content)).convert("RGB").resize((768, 768))

        mask = np.zeros((768, 768), dtype=np.float32)
        mask[0:160, :] = 1.0

        for y in range(160, 260):
            mask[y, :] = 1.0 - ((y - 160) / 100)

        mask = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask)

        result = pipe(
            prompt=f"professional haircut, {prompt}, realistic hair",
            image=img,
            mask_image=mask_img,
            strength=0.8,
            guidance_scale=8.5,
            num_inference_steps=25
        ).images[0]

        buf = io.BytesIO()
        result.save(buf, format="JPEG")

        return {"result_image": base64.b64encode(buf.getvalue()).decode()}

    except Exception as e:
        return {"error": str(e)}

# =========================================
# 📩 WAITLIST
# =========================================
@app.post("/waitlist")
async def waitlist(request: Request):
    try:
        email = None

        # JSON
        try:
            data = await request.json()
            email = data.get("email")
        except:
            pass

        # FORM fallback
        if not email:
            form = await request.form()
            email = form.get("email")

        if not email:
            return {"error": "email missing"}

        # SAVE
        with open("/var/www/html/subscribers.txt", "a") as f:
            f.write(f"{email} | {datetime.datetime.now()}\n")

        # 🔥 EMAIL FLOW
        send_admin_notification(email)
        send_user_receipt(email)

        return {"status": "saved"}

    except Exception as e:
        return {"error": str(e)}

# =========================================
# 🚀 START
# =========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
