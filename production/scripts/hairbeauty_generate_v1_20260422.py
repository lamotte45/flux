# RUN THIS SCRIPT WITH:
# python3 /home/kenny/barber_ai/scripts/test_hairbeauty_cycle_v2.py

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import os

OUTPUT_DIR = "/home/kenny/barber_ai/test_outputs/hairbeauty_cycle_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# caption log file
log_path = os.path.join(OUTPUT_DIR, "captions.txt")
log = open(log_path, "w")

# Global deterministic seed
SEED = 12345
generator = torch.Generator("cuda").manual_seed(SEED)

# Back‑view deterministic seeds
backview_seeds = {
    "HairBeauty, womens hairstyle, back view, long straight hair with smooth ends, black woman, white background": 9001,
    "HairBeauty, natural afro style, back view, rounded afro shape with clean outline, black woman, white background": 9002,
    "HairBeauty, hair color style, back view, blonde balayage with smooth layers, woman, white background": 9003,
}

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("/home/kenny/barber_ai/lora_outputs/hairbeauty_v1/hairbeauty_v1.safetensors")
pipe.fuse_lora(lora_scale=0.8)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

categories = {
    "braids": [
        "HairBeauty, braids style, medium knotless braids with clean parts, black woman, white background",
        "HairBeauty, braids style, feed-in cornrows with curved pattern, black woman, white background",
        "HairBeauty, braids style, braided ponytail with scalp design, black woman, white background",
    ],
    "hair_color": [
        "HairBeauty, hair color style, bright copper orange hair, woman, white background",
        "HairBeauty, hair color style, icy silver balayage, woman, white background",
        "HairBeauty, hair color style, teal blue ombre hair, woman, white background",
        "HairBeauty, hair color style, back view, blonde balayage with smooth layers, woman, white background",
    ],
    "natural_afro": [
        "HairBeauty, natural afro style, tight coil afro with clean defined texture, black woman, white background",
        "HairBeauty, natural afro style, shoulder-length twist-out with uniform curl pattern, black woman, white background",
        "HairBeauty, natural afro style, tapered afro cut with neat sponge curls, black woman, white background",
        "HairBeauty, natural afro style, back view, rounded afro shape with clean outline, black woman, white background",
    ],
    "beard": [
        "HairBeauty, beard style, sharp mid-fade with full beard lineup, african american man, white background",
        "HairBeauty, beard style, short boxed beard with crisp cheek line, african american man, white background",
        "HairBeauty, beard style, chin strap beard with clean taper, african american man, white background",
    ],
    "womens_styles": [
        "HairBeauty, womens hairstyle, layered bob with side part, black woman, white background",
        "HairBeauty, womens hairstyle, voluminous curls with middle part, black woman, white background",
        "HairBeauty, womens hairstyle, silk press with smooth ends, black woman, white background",
        "HairBeauty, womens hairstyle, back view, long straight hair with smooth ends, black woman, white background",
    ],
}

NEG = "blurry, bad quality, watermark, text, dark background, mannequin, 3d render"

index = 0

for category, prompts in categories.items():
    for prompt in prompts:
        index += 1
        print(f"Generating {index}: {category}...")

        # Choose deterministic seed for back‑view prompts
        if prompt in backview_seeds:
            local_seed = backview_seeds[prompt]
            local_gen = torch.Generator("cuda").manual_seed(local_seed)
        else:
            local_gen = generator

        image = pipe(
            prompt=prompt,
            negative_prompt=NEG,
            num_inference_steps=25,
            guidance_scale=7.0,
            width=1024,
            height=1024,
            generator=local_gen
        ).images[0]

        filename = f"{category}_{index:02d}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        image.save(save_path)

        # log caption
        log.write(f"{filename} : {prompt}\n")

        print(f"Saved {filename}")

log.close()
print(f"Done! Check {OUTPUT_DIR}")
