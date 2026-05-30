import gc
import uuid
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch

from tsr.system import TSR

# 1. Mesh output directory
MESH_DIR = Path("/tmp/barber_ai_meshes")
MESH_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# 2. Expose /files so Nginx can proxy it
app.mount("/files", StaticFiles(directory=str(MESH_DIR)), name="files")

# 3. Load Stable Fast 3D / TripoSR model
print("🚀 Loading TripoSR model...")
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt"
)
model.to("cuda:0")
model.eval()
print("✅ Model loaded")

@app.post("/reconstruct")
async def reconstruct(image: UploadFile = File(...)):
    try:
        # Read the uploaded image into memory
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")

        # Clear VRAM
        gc.collect()
        torch.cuda.empty_cache()

        # Run 3D reconstruction
        with torch.no_grad():
            scene_codes = model([img], device="cuda:0")
            mesh = model.extract_mesh(
                scene_codes,
                has_vertex_color=False,
                resolution=256
            )[0]

        # Unique mesh id
        mesh_id = str(uuid.uuid4())
        mesh_path = MESH_DIR / f"{mesh_id}.obj"

        # Export OBJ
        mesh.export(str(mesh_path), file_type="obj")
        print(f"✅ Saved mesh to: {mesh_path}")

        # Public URL your app will download from
        mesh_url = f"https://aibeautyconcepts.com/files/{mesh_id}.obj"

        return {
            "status": "success",
            "mesh_url": mesh_url
        }

    except Exception as e:
        print(f"❌ 3D Error: {e}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
