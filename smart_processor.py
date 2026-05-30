import os
import torch
import trimesh
import cv2
import numpy as np
from PIL import Image, ImageOps
from tsr.system import TSR
from tsr.utils import remove_background

# Environment Setup
os.environ["ORT_DISABLE_GPU"] = "1"
torch.cuda.empty_cache()

print("🔥 Loading TripoSR...")
model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt").to("cuda").eval()

def process_smart():
    input_path = "/home/user/barber_ai/dataset_v2/concept/10_platinumwaves/ChatGPT Image Mar 15, 2026, 07_45_13 PM.png"
    output_path = "/home/user/barber_ai/smart_platinum.glb"
    
    print(f"📸 Loading: {input_path}")
    image = Image.open(input_path).convert("RGB")
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("🔍 Detecting face...")
    faces = face_cascade.detectMultiScale(cv_img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        print("✅ Face detected — applying smart crop")
        (x, y, w, h) = faces[0]
        top = max(0, y - int(h * 0.50))
        bottom = min(image.height, y + int(h * 1.20))
        left = max(0, x - int(w * 0.25))
        right = min(image.width, x + int(w * 1.25))
        image = image.crop((left, top, right, bottom))
    else:
        print("⚠️ No face detected — using fallback")
        image = image.crop((0, 0, image.width, int(image.height * 0.70)))

    print("🧱 Normalizing image...")
    image = ImageOps.expand(image, border=(0, 60, 0, 0), fill=(255, 255, 255))
    image = ImageOps.autocontrast(image)

    print("✂️ Removing background...")
    image_no_bg = remove_background(image).convert("RGBA")
    clean = Image.new("RGB", image_no_bg.size, (255, 255, 255))
    clean.paste(image_no_bg, mask=image_no_bg.split()[3])

    print("🧠 Building 3D Mesh...")
    with torch.no_grad():
        scene_codes = model([clean], device="cuda")
        mesh = model.extract_mesh(scene_codes, resolution=256, threshold=30.0, has_vertex_color=True)[0]

    # Orientation & Centering
    rotation = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    mesh.apply_transform(rotation)
    mesh.apply_translation(-mesh.centroid)
    
    # Scale Normalization
    bounds = mesh.bounds
    mesh.apply_scale(1.0 / (bounds[1][2] - bounds[0][2]))
    mesh.apply_translation(-mesh.centroid)

    mesh.export(output_path)
    print(f"✅ SUCCESS: {output_path}")

process_smart()
