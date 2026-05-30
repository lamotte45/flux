import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import os
from pathlib import Path

# Load the inpainting brain - using float16 for speed on your 4090
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

input_dir = Path("/home/user/barber_ai/raw_photos")
output_dir = Path("/home/user/barber_ai/training_data/healed")
output_dir.mkdir(parents=True, exist_ok=True)

print("🎨 Starting Generative Outpainting for [AI Beauty Concepts]...")

for img_p in input_dir.glob("*.[jp][pn]g"):
    # Load and prep
    init_image = Image.open(img_p).convert("RGB").resize((1024, 1024))
    
    # Create expanded canvas (adding 200px to top)
    width, height = init_image.size
    new_height = height + 200
    expanded_image = Image.new("RGB", (width, new_height), (0, 0, 0))
    expanded_image.paste(init_image, (0, 200))
    
    # Create the mask (tell AI to ONLY paint the top 200px)
    mask_image = Image.new("RGB", (width, new_height), (255, 255, 255))
    mask_image.paste(Image.new("RGB", (width, height), (0, 0, 0)), (0, 200))
    
    # Specific prompt for your hair textures
    prompt = "top of voluminous kinky straight hair, detailed 4C hair texture, highly detailed, realistic skin, matching background"
    
    # Run the 4090
    with torch.inference_mode():
        healed_image = pipe(
            prompt=prompt, 
            image=expanded_image, 
            mask_image=mask_image,
            num_inference_steps=30
        ).images[0]
    
    # Crop back to a perfect 1024x1024 square but shifted up
    final_crop = healed_image.crop((0, 0, 1024, 1024))
    
    out_name = output_dir / f"healed_{img_p.name}"
    final_crop.save(out_name)
    print(f"✅ Healed and Outpainted: {img_p.name}")

print(f"\n🚀 Success! Your improved dataset is in {output_dir}")
