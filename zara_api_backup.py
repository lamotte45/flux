
# -*- coding: utf-8 -*-
import os
import base64
import json
import boto3
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from anthropic import Anthropic

app = FastAPI()

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)

MODEL_ID = "stability.sd3-5-large-v1:0"

def generate_selfie_style(selfie_b64: str, prompt: str) -> str:
    body = json.dumps({
        "prompt": prompt,
        "mode": "image-to-image",
        "image": selfie_b64,
        "strength": 0.65,
        "output_format": "jpeg"
    })

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=body,
        accept="application/json",
        contentType="application/json"
    )

    result = json.loads(response["body"].read())

    if "images" in result and len(result["images"]) > 0:
        return result["images"][0]

    if "artifacts" in result and len(result["artifacts"]) > 0:
        return result["artifacts"][0].get("base64")

    return None

def detect_category(text: str) -> str:
    t = text.lower()
    if "fade" in t or "lineup" in t or "waves" in t:
        return "barbershop"
    if "afro" in t or "coils" in t:
        return "afros"
    if "beard" in t or "goatee" in t:
        return "beards"
    if "braids" in t or "locs" in t or "twists" in t:
        return "braids"
    return "women"

ZARA_SYSTEM_PROMPT = "You are Zara, a friendly AI stylist."

claude = Anthropic()

@app.get("/health")
def health():
    return {"status": "ok", "service": "zara_api"}

@app.post("/avatar/request_with_selfie")
async def avatar_request_with_selfie(
    text: str = Form(...),
    selfie: UploadFile = File(...),
    fade_type: str = Form("taper"),
    num_images: int = Form(1)
):
    try:
        selfie_bytes = await selfie.read()
        selfie_b64 = base64.b64encode(selfie_bytes).decode()

        hairstyle_prompt = f'''
        Apply the hairstyle described here to the person in the selfie:
        {text}

        Keep the person's face, identity, and skin tone unchanged.
        Only modify the hair.
        '''

        try:
            output_b64 = generate_selfie_style(selfie_b64, hairstyle_prompt)
            images = [output_b64] if output_b64 else []
        except Exception:
            images = []

        if not images:
            category = detect_category(text)
            catalog_dir = f"/home/kenny/barber_ai/catalog/{category}"
            images = []
            if os.path.exists(catalog_dir):
                for img in sorted(os.listdir(catalog_dir))[:num_images]:
                    if img.endswith(".jpg"):
                        with open(os.path.join(catalog_dir, img), "rb") as f:
                            images.append(base64.b64encode(f.read()).decode())

        response = claude.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            system=ZARA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}]
        )

        zara_text = response.content[0].text.replace("*", "").replace("#", "").strip()

        return JSONResponse({
            "status": "ok",
            "emotion": "happy",
            "avatar_message": zara_text,
            "images": images,
            "category": detect_category(text)
        })

    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)
