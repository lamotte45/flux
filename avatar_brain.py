import httpx
import uuid
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI()

GEOMETRY_URL = "http://127.0.0.1:8000/reconstruct"
STYLIST_URL = "http://127.0.0.1:8001/generate_styles"

@app.post("/avatar/request")
async def process_avatar(image: UploadFile = File(...), text: str = Form(...)):
    trace_id = str(uuid.uuid4())[:8]
    try:
        # Read file once
        img_bytes = await image.read()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 1. Get Mask from Geometry
            print(f"[TRACE-{trace_id}] Requesting Mask...")
            geo_res = await client.post(
                GEOMETRY_URL, 
                files={"image": ("file.jpg", img_bytes, "image/jpeg")}
            )
            
            if geo_res.status_code != 200:
                return JSONResponse(status_code=500, content={"error": "3D Masking Failed"})

            # 2. Send to Stylist
            print(f"[TRACE-{trace_id}] Sending to Stylist...")
            # Use geo_res.content if mask is working, or img_bytes for a direct test
            style_res = await client.post(
                STYLIST_URL,
                files={"image": ("file.jpg", img_bytes, "image/jpeg")},
                data={"prompt": text}
            )

            return style_res.json()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
