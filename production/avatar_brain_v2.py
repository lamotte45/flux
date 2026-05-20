import sys
import time
import traceback
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

if "/home/kenny/barber_ai" not in sys.path:
    sys.path.insert(0, "/home/kenny/barber_ai")

app = FastAPI(title="Avatar Brain v2")

@app.get("/")
async def root():
    return {"status": "ok", "service": "avatar_brain_v2"}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "avatar_brain_v2"}

@app.post("/avatar/request_with_selfie")
async def avatar_request_with_selfie(
    text: str = Form(...),
    fade_type: str = Form("taper"),
    selfie: UploadFile = File(...),
):
    try:
        from selfie_inpaint_v5 import run_selfie_inpaint
    except Exception as e:
        return JSONResponse({"status": "error", "error": f"Import failed: {e}", "trace": traceback.format_exc(), "images": [], "image_count": 0}, status_code=500)
    try:
        selfie_bytes = await selfie.read()
        FADE_PROMPTS = {
            'bald_fade': 'bald fade haircut, skin fade sides, natural curly hair on top, same person same face, realistic barbershop photo',
            'low_fade': 'low fade haircut, natural hair on top, tapered sides, same person same face, realistic barbershop photo',
            'mid_fade': 'mid fade haircut, natural hair on top, clean sides, same person same face, realistic barbershop photo',
            'high_fade': 'high fade haircut, natural hair on top, shaved sides, same person same face, realistic barbershop photo',
            'taper': 'taper fade haircut, natural hair on top, clean neckline, same person same face, realistic barbershop photo',
            'saints_fade': 'saints fade haircut, low fade, natural curly hair on top, same person same face, realistic barbershop photo',
            'lineup': 'sharp lineup, edge up, natural hair, same person same face, realistic barbershop photo',
        }
        prompt = FADE_PROMPTS.get(fade_type, f"{text}, fade haircut, same person same face, realistic barbershop photo")
        start = time.perf_counter()
        b64, mode = run_selfie_inpaint(selfie_bytes, prompt, fade_type=fade_type)
        elapsed = round(time.perf_counter() - start, 2)
        return JSONResponse({"status": "ok", "style": fade_type, "mode": mode, "inference_time": elapsed, "action": "show_previews", "images": [b64], "image_count": 1})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e), "trace": traceback.format_exc(), "images": [], "image_count": 0}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
