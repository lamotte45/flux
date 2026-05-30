import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline

# --- IMPORT YOUR NEW GUARDRAILS ---
from hair_negatives import NEGATIVE_HAIR, POSITIVE_HAIR

# ---------------- CONFIG ----------------
LORA_PATH = "/home/user/barber_ai/models/lora/ABC_KinkyStraight_v3.safetensors"
RAW_DIR = Path("/home/user/barber_ai/raw_photos")
OUT_DIR = Path("/home/user/barber_ai/outputs/final_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
).to(device)
pipe.load_lora_weights(LORA_PATH)

yolo = YOLO("yolov8n.pt")
sam = sam_model_registry["vit_h"](checkpoint="/home/user/models/sam/sam_vit_h_4b8939.pth").to(device)
sam_predictor = SamPredictor(sam)

# ---------------- PROCESS ----------------
for img_path in RAW_DIR.glob("*.*"):
    print(f"🖼 Applying Anti-Bald Pipeline to: {img_path.name}")
    rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    results = yolo(rgb, verbose=False)
    if not results[0].boxes: continue
    x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)

    # Padding for hair height
    head_h = y2 - y1
    y1_new = max(0, y1 - int(head_h * 0.45)) # Expand UP for big hair
    
    sam_predictor.set_image(rgb)
    input_box = np.array([x1, y1_new, x2, y2])
    masks, scores, _ = sam_predictor.predict(box=input_box, multimask_output=True)
    mask = (masks[np.argmax(scores)].astype(np.uint8) * 255)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    init_img = Image.fromarray(cv2.resize(rgb, (1024, 1024)))
    mask_img = Image.fromarray(cv2.resize(mask, (1024, 1024)))

    # COMBINED PROMPT
    prompt = (
        f"studio portrait of a woman, abcstyle kinky straight hair, {POSITIVE_HAIR}, "
        f"long hair, thick hair volume, natural black color, highly detailed strands"
    )

    negative = (
        f"{NEGATIVE_HAIR}, jewelry, earrings, jewelry, man, distorted, cgi"
    )

    output = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=init_img,
        mask_image=mask_img,
        num_inference_steps=50,
        strength=0.82,          # High strength to force hair over bald areas
        guidance_scale=14.0,     # Strong guidance for strict prompt adherence
        cross_attention_kwargs={"scale": 0.85}
    ).images[0]

    output.save(OUT_DIR / f"anti_bald_{img_path.stem}.png")
    print(f"✅ Success: anti_bald_{img_path.name}")
