import cv2
import torch
import numpy as np
import random
from pathlib import Path
from PIL import Image

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from controlnet_aux import MidasDetector
from gfpgan import GFPGANer

# ---------------- CONFIG & PATHS ----------------
DEVICE = "cuda"
MODEL_DIR = Path("/home/user/barber_ai/models")
RAW_DIR = Path("/home/user/barber_ai/raw_photos")
OUT_DIR = Path("/home/user/barber_ai/outputs/pro_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Update this once you have your SDXL-specific LoRA
LORA_PATH = "/home/user/barber_ai/models/lora/ABC_KinkyStraight_SDXL.safetensors"

# ---------------- LOAD MODELS ----------------
print("📦 Loading SDXL + ControlNet + IP-Adapter...")
controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-depth-sdxl-1.0", 
    torch_dtype=torch.float16
).to(DEVICE)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(DEVICE)

# FIX #3 & #8: IP-Adapter & Optimizations
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
pipe.set_ip_adapter_scale(0.7)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_tiling()

# Load Support Models
yolo = YOLO("yolov8n.pt")
sam = sam_model_registry["vit_h"](checkpoint="/home/user/models/sam/sam_vit_h_4b8939.pth").to(DEVICE)
sam_predictor = SamPredictor(sam)
depth_estimator = MidasDetector.from_pretrained("Intel/dpt-hybrid-midas")
face_restorer = GFPGANer(model_path=str(MODEL_DIR/"face_restore/GFPGANv1.4.pth"), upscale=1)

# ---------------- PROCESSING ----------------
styles = ["kinky straight silk press", "voluminous salon blowout", "layered natural texture"]

for img_path in RAW_DIR.glob("*.*"):
    print(f"✨ Generating Pro Preview for: {img_path.name}")
    
    # Load and Detect
    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # YOLO & SAM (Simplified as per Fix #5)
    results = yolo(rgb, verbose=False)
    x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    
    sam_predictor.set_image(rgb)
    masks, scores, _ = sam_predictor.predict(box=np.array([x1, y1-40, x2, y2]), multimask_output=True)
    mask_binary = masks[np.argmax(scores)]
    mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8)).convert("L")

    # FIX #4: Depth Map Handling
    init_image = Image.fromarray(rgb).resize((1024, 1024))
    depth_image = depth_estimator(init_image)

    # FIX #1: Random Seed Variation
    gen = torch.Generator(DEVICE).manual_seed(random.randint(0, 2**32 - 1))

    # FIX #2 & #6: SDXL Generation Call
    prompt = f"studio portrait of a beautiful woman, abcstyle hair, {random.choice(styles)}, realistic strands, 8k"
    negative = "bald, jewelry, earrings, distorted, cartoon, cgi, lowres"

    output = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=init_image,
        controlnet_conditioning_image=depth_image, # Correct SDXL Param
        ip_adapter_image=init_image,              # Identity Lock
        mask_image=mask_pil.resize((1024,1024)),
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=gen,
        cross_attention_kwargs={"scale": 0.5}     # LoRA Strength
    ).images[0]

    # FIX #6: Face Restore
    out_np = np.array(output)
    restored_img, _, _ = face_restorer.enhance(out_np, has_aligned=False, only_center_face=True, paste_back=True)
    
    # Save Final
    final_out = Image.fromarray(restored_img)
    final_out.save(OUT_DIR / f"pro_{img_path.stem}.png")

print(f"✅ Success! Results saved in {OUT_DIR}")
