import sys
import boto3
import json
import base64
import time
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, '/home/kenny/barber_ai/production')
sys.path.insert(0, '/home/kenny/barber_ai')

from zara_brain.prompts import ZARA_SYSTEM_PROMPT, detect_category

app = FastAPI(title="Zara AI Concierge")

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

@app.get("/")
async def root():
    return {"status": "online", "msg": "Zara is ready"}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "zara_api"}

@app.post("/zara/chat")
async def zara_chat(text: str = Form(...)):
    try:
        # Get Zara's response from Claude via Bedrock
        response = bedrock.invoke_model(
            modelId="arn:aws:bedrock:us-east-1:852736927991:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "system": ZARA_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": text}]
            })
        )
        result = json.loads(response['body'].read())
        zara_text = result['content'][0]['text']
        category = detect_category(text)
        return JSONResponse({
            "status": "ok",
            "response": zara_text,
            "category": category,
            "styles_url": f"/styles/{category}"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@app.post("/zara/recommend")
async def zara_recommend(text: str = Form(...)):
    category = detect_category(text)
    # Load styles from catalog
    import os
    catalog_dir = f"/home/kenny/barber_ai/catalog/{category}"
    styles = []
    if os.path.exists(catalog_dir):
        for img in sorted(os.listdir(catalog_dir))[:6]:
            if img.endswith('.jpg'):
                style_name = img.replace('.jpg','').replace(f'{category}_','')
                with open(os.path.join(catalog_dir, img), 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
                styles.append({
                    "id": img.replace('.jpg',''),
                    "name": style_name.replace('_',' ').title(),
                    "image": b64
                })
    return JSONResponse({
        "status": "ok",
        "category": category,
        "styles": styles
    })

@app.get("/avatar")
async def get_avatar():
    avatar_path = "/home/kenny/barber_ai/production/avatar_base.jpg"
    with open(avatar_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return JSONResponse({"status": "ok", "image": b64})


@app.get("/zara/clips")
async def get_clips():
    return JSONResponse({
        "clips": {
            "welcome": "/static/zara_clips/welcome.mp4",
            "expressions": "/static/zara_clips/expressions.mp4"
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
