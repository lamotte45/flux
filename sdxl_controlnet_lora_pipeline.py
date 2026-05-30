import torch
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import cv2
import os

# 1. Load ControlNet (Canny) - This will download ~5GB
print("📥 Downloading/Loading ControlNet Canny...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

# 2. Load SDXL Base - This will download ~12GB
print("📥 Downloading/Loading SDXL Base Inpaint...")
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# 3. Load the LoRA Suite (Style, Texture, Color)
print("🧠 Loading your custom Barber LoRAs...")
lora_path = "/home/user/barber_ai/models/lora/"
pipe.load_lora_weights(os.path.join(lora_path, "ABC_Style_SDXL_v1.safetensors"), adapter_name="style")
pipe.load_lora_weights(os.path.join(lora_path, "ABC_Texture_SDXL_v1.safetensors"), adapter_name="texture")
pipe.load_lora_weights(os.path.join(lora_path, "ABC_Color_SDXL_v1.safetensors"), adapter_name="color")

pipe.set_adapters(
    ["style", "texture", "color"],
    adapter_weights=[0.8, 0.7, 0.4]
)

# 4. Process Images
print("🖼️ Processing images and generating edges...")
image = load_image("client_photo.png").convert("RGB")
mask = load_image("sam_hair_mask.png").convert("L") # Ensure mask is Grayscale

# Create Canny edges from the original image for guidance
image_np = np.array(image)
canny_image = cv2.Canny(image_np, 100, 200)
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image] * 3, axis=2)
control_image = Image.fromarray(canny_image)

# 5. Run the Install
print("🚀 Running the AI Hair Install...")
output = pipe(
    prompt="professional barber install, KinkyStraight hair texture, sharp fade, highly detailed, 8k",
    negative_prompt="blurry, distorted, low quality, messy hairline, cartoonish",
    image=image,
    mask_image=mask,
    control_image=control_image,
    controlnet_conditioning_scale=1.1,
    strength=0.8,
    num_inference_steps=40
).images[0]

output.save("final_client_result.png")
print("✨ Finished! Saved to: final_client_result.png")
