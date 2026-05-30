import torch
import uuid
import os
import io
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from rembg import remove

app = FastAPI()

# Setup paths
MESH_DIR = Path("/home/user/barber_ai/static/files")
MESH_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(MESH_DIR)), name="files")

print("🚀 Loading 3D Geometry Engine (TripoSR)...")
# Note: In a full production env, you'd load the TripoSR model here
# model = TSR.from_pretrained("stabilityai/TripoSR", ...)

@app.post("/reconstruct")
async def reconstruct(image: UploadFile = File(...)):
    try:
        mesh_id = str(uuid.uuid4())
        contents = await image.read()
        input_img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # --- STEP 1: Background Removal (Essential for clean 3D) ---
        print(f"⌛ Removing background for ID: {mesh_id}")
        # rembg handles the alpha transparency automatically
        no_bg_img = remove(input_img)
        
        # --- STEP 2: TripoSR Reconstruction ---
        # scene_codes = model([no_bg_img], device="cuda:0")
        # mesh = model.extract_mesh(scene_codes)[0]
        
        # Export logic
        file_name = f"{mesh_id}.obj"
        file_path = MESH_DIR / file_name
        
        # For now, we save the processed image to verify the 'rembg' works
        # In final: mesh.export(str(file_path), file_type="obj")
        no_bg_img.save(f"/home/user/barber_ai/static/files/{mesh_id}_check.png")
        
        with open(file_path, "w") as f:
            f.write("# Real TripoSR Mesh Data would be here")

        return JSONResponse(content={
            "status": "success", 
            "mesh_url": f"https://aibeautyconcepts.com/files/{file_name}",
            "check_url": f"https://aibeautyconcepts.com/files/{mesh_id}_check.png"
        })

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
