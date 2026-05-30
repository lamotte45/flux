import os
import torch
import trimesh
import numpy as np
from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background

# 1. Environment & Memory Setup
os.environ["ORT_DISABLE_GPU"] = "1"
torch.cuda.empty_cache()

print("🔥 Loading TripoSR...")
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt"
).to("cuda").eval()

model.renderer.chunk_size = 4096

def process_platinum_waves():
    input_path = "/home/user/barber_ai/dataset_v2/concept/10_platinumwaves/ChatGPT Image Mar 15, 2026, 07_45_13 PM.png"
    # We will save it to a clear, specific name
    output_path = "/home/user/barber_ai/platinum_final.glb"
    
    if not os.path.exists(input_path):
        print(f"❌ Error: Image not found at {input_path}")
        return

    print(f"📸 Opening: {input_path}")
    image = Image.open(input_path).convert("RGB")

    # --- THE FIX: CROP SHOULDERS (Top 70% only) ---
    width, height = image.size
    image = image.crop((0, 0, width, int(height * 0.70)))

    print("✂️ Removing background...")
    image_no_bg = remove_background(image).convert("RGBA")

    clean = Image.new("RGB", image_no_bg.size, (255, 255, 255))
    clean.paste(image_no_bg, mask=image_no_bg.split()[3])

    print("🧠 Building 3D mesh...")
    with torch.no_grad():
        scene_codes = model([clean], device="cuda")
        mesh = model.extract_mesh(
            scene_codes,
            resolution=256,
            threshold=30.0, # High threshold to clean the edges
            has_vertex_color=True
        )[0]

    print("🧭 Standing it up & centering...")
    rotation = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    mesh.apply_transform(rotation)
    mesh.apply_translation(-mesh.centroid)

    # Normalize Scale
    bounds = mesh.bounds
    mesh.apply_scale(1.0 / (bounds[1][2] - bounds[0][2]))
    mesh.apply_translation(-mesh.centroid)

    mesh.export(output_path)
    print(f"✅ DONE → {output_path}")

if __name__ == "__main__":
    process_platinum_waves()
