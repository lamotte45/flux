import os
import cv2
import torch
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter
from ultralytics import YOLO
from diffusers import StableDiffusionXLInpaintPipeline

YOLO_MODEL = "/home/kenny/barber_ai/models/yolov8n-face.pt"
SDXL_MODEL = "/home/kenny/barber_ai/models/sdxl"
LORA_PATH = "/home/kenny/barber_ai/lora_outputs/razor_sdxl_v3/razor_sdxl_v3.safetensors"

FADE_STYLES = {"fade","taper","low_fade","mid_fade","high_fade","bald_fade","lineup","saints_fade"}

_device = "cuda" if torch.cuda.is_available() else "cpu"
_yolo = None
_pipe = None

def get_yolo():
    global _yolo
    if _yolo is None:
        _yolo = YOLO(YOLO_MODEL)
    return _yolo

def get_inpaint_pipe():
    global _pipe
    if _pipe is None:
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            SDXL_MODEL + "/sd_xl_base_1.0.safetensors",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(_device)
        if os.path.exists(LORA_PATH):
            pipe.load_lora_weights(LORA_PATH)
            pipe.fuse_lora(lora_scale=0.7)
        pipe.set_progress_bar_config(disable=True)
        _pipe = pipe
    return _pipe

def build_fade_mask(img_np, x1, y1, x2, y2):
    h, w = img_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    face_h = y2 - y1
    face_w = x2 - x1
    # Top hair area
    hair_top = max(0, y1 - int(face_h * 0.4))
    cv2.rectangle(mask, (x1, hair_top), (x2, y1), 255, -1)
    # Left side - stop at jaw level
    cv2.rectangle(mask, (max(0, x1 - int(face_w * 0.8)), int(y1 + face_h * 0.02)), (x1, int(y2 - face_h * 0.02)), 255, -1)
    # Right side - stop at jaw level
    cv2.rectangle(mask, (x2, int(y1 + face_h * 0.02)), (min(w, x2 + int(face_w * 0.8)), int(y2 - face_h * 0.02)), 255, -1)
    # Nape - below face center
    cv2.rectangle(mask, (x1, y2), (x2, min(h, y2 + int(face_h * 0.2))), 255, -1)
    return mask

def build_full_mask(img_np, x1, y1, x2, y2):
    h, w = img_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    face_h = y2 - y1
    face_w = x2 - x1
    hair_top = max(0, y1 - int(face_h * 1.4))
    hair_x1 = max(0, x1 - int(face_w * 0.8))
    hair_x2 = min(w, x2 + int(face_w * 0.8))
    cv2.rectangle(mask, (hair_x1, hair_top), (hair_x2, y1 + int(face_h * 0.15)), 255, -1)
    cv2.rectangle(mask, (hair_x1, y1), (x1, y2), 255, -1)
    cv2.rectangle(mask, (x2, y1), (hair_x2, y2), 255, -1)
    return mask

def detect_hair_mask(image_pil, mode="full"):
    yolo = get_yolo()
    img_np = np.array(image_pil.convert("RGB"))
    h, w = img_np.shape[:2]
    results = yolo(img_np, conf=0.25, verbose=False)
    mask_np = np.zeros((h, w), dtype=np.uint8)
    found = False
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            if mode == "fade":
                mask_np = build_fade_mask(img_np, x1, y1, x2, y2)
            else:
                mask_np = build_full_mask(img_np, x1, y1, x2, y2)
            found = True
            break
        if found:
            break
    if not found:
        mask_np[0:int(h * 0.55), :] = 255
    mask = Image.fromarray(mask_np)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=12))
    return mask

def run_selfie_inpaint(selfie_bytes, prompt, neg_prompt="blurry, distorted face, extra limbs, bad anatomy", fade_type="taper", steps=30):
    mode = "fade" if fade_type in FADE_STYLES else "full"
    strength = 0.75 if mode == "fade" else 0.90
    image = Image.open(BytesIO(selfie_bytes)).convert("RGB")
    image = image.resize((1024, 1024))
    mask = detect_hair_mask(image, mode=mode)
    pipe = get_inpaint_pipe()
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            guidance_scale=8.5,
            num_inference_steps=steps
        ).images[0]
    buf = BytesIO()
    result.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), mode

if __name__ == "__main__":
    print("Selfie Inpaint v5 - Dual Mode OK")
