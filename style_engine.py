from fastapi import FastAPI, UploadFile, File, Form
from diffusers import AutoPipelineForInpainting
from PIL import Image
import torch, io, base64, numpy as np

app = FastAPI()

pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_attention_slicing()

print("🔥 STYLE ENGINE READY (FEATHERED MASK)")

def encode(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

@app.post("/generate_styles")
async def generate(image: UploadFile = File(...), prompt: str = Form(...)):
    try:
        content = await image.read()
        img = Image.open(io.BytesIO(content)).convert("RGB").resize((512, 512))

        # 🔥 FEATHERED HAIR MASK (NO HARD LINE)
        mask = np.zeros((512, 512), dtype=np.float32)

        # strong top area
        mask[0:140, :] = 1.0

        # smooth blend zone
        for y in range(140, 220):
            mask[y, :] = 1.0 - ((y - 140) / 80)

        mask = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask)

        # 🔥 DYNAMIC PROMPT
        final_prompt = f"""
        professional hairstyle photo, {prompt},
        clean haircut, detailed hair texture,
        natural hairline, realistic scalp transition,
        preserve same face, studio lighting
        """

        # 🔥 NEGATIVE PROMPT (REDUCE ARTIFACTS)
        negative_prompt = """
        hard hairline, wig, helmet hair, floating hair,
        distorted face, unrealistic, blurry, artifacts
        """

        result = pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            image=img,
            mask_image=mask_img,
            strength=0.8,
            guidance_scale=8.5,
            num_inference_steps=25
        ).images[0]

        return {
            "style_image": encode(result)
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
