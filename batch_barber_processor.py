import torch, os, glob, cv2
import numpy as np
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image, ImageFilter

out_dir = "barber_results_rugged_natural"
os.makedirs(out_dir, exist_ok=True)
lora_dir = "/home/user/barber_ai/models/lora/"

print("🚀 Starting RUGGED NATURAL Production Run (No Makeup/Cartoon)...")
controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights(lora_dir + "ABC_Style_SDXL_v1.safetensors", adapter_name="style")
pipe.load_lora_weights(lora_dir + "ABC_Texture_SDXL_v1.safetensors", adapter_name="texture")
pipe.load_lora_weights(lora_dir + "ABC_Color_SDXL_v1.safetensors", adapter_name="color")
pipe.set_adapters(["style", "texture", "color"], adapter_weights=[0.8, 0.7, 0.4])

images = sorted([f for f in glob.glob("*.png") if all(x not in f for x in ["res_", "mask", "v2_", "v3_"])])

for img_p in images:
    print(f"💈 Processing Natural Male Textures: {img_p}")
    img = load_image(img_p).convert("RGB")
    mask = Image.new("L", img.size, 255).filter(ImageFilter.GaussianBlur(radius=12)) 
    canny = cv2.Canny(np.array(img), 100, 200)
    ctrl = Image.fromarray(np.stack([canny]*3, axis=-1))
    
    res = pipe(
        # THE POSITIVE: Adding "Skin pores" and "Masculine" forces realism
        prompt="professional masculine barber install, intricate platinum waves, natural masculine skin texture, visible skin pores, rugged, realistic lighting, highly detailed 8k",
        # THE NEGATIVE: Banning makeup and cartoonish features
        negative_prompt="makeup, lipstick, eyeliner, mascara, feminine features, smooth plastic skin, cartoon, anime, illustration, painting, 3d render, doll, airbrushed, feminine eyes",
        image=img, mask_image=mask, control_image=ctrl,
        num_inference_steps=30, 
        strength=0.62 # Slightly lower strength to keep the original man's face 100% real
    ).images[0]
    res.save(f"{out_dir}/rugged_{img_p}")

print("✨ RUGGED BATCH COMPLETE!")
