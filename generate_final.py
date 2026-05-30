import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import os

LORA_PATH = "/home/kenny/barber_ai/lora_outputs/razor_sdxl_v3"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = "/home/kenny/barber_ai/test_outputs/final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=0.9)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

NEGATIVE = (
    "tattoo, ink, blurry, cartoon, metallic, jewelry, floating object, "
    "busy background, brick wall, barbershop background, cluttered background, "
    "colorful background, text overlay, watermark, shirt design, logo on clothing, "
    "patterned clothing, suit, fashion, editorial, plastic skin, dark skin, monochrome, black and white"
)


def generate(design_name: str, prompt: str, seed: int = 42):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024,
        generator=generator,
    ).images[0]
    path = os.path.join(OUTPUT_DIR, f"{design_name}.png")
    image.save(path)
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    kobe_dir = "/home/kenny/barber_ai/test_outputs/kobe_attempt"
    os.makedirs(kobe_dir, exist_ok=True)
    kobe_prompt = "RazorArtStyle, ultra realistic barber haircut design, african american man, side profile, clean low skin fade, intricate portrait of Kobe Bryant shaved into the fade haircut, hyper-detailed hair engraving, razor-etched design using natural hair texture, basketball and KOBE text carved into hair, number 24 integrated into the design, perfect shading using hair density, stippling technique, smooth gradient transitions, high contrast between shaved and unshaved hair, design embedded only in the fade area, natural curls on top, sharp lineup, crisp edges, professional barbershop quality, photorealistic skin texture, DSLR lighting, 85mm lens, ultra sharp focus, 8k detail, no overlay, no sticker effect, design looks physically carved into hair"
    for i in range(4):
        generator = torch.Generator(device="cuda").manual_seed(42 + i)
        image = pipe(prompt=kobe_prompt, negative_prompt=NEGATIVE, num_inference_steps=30, guidance_scale=7.5, width=1024, height=1024, generator=generator).images[0]
        path = os.path.join(kobe_dir, f"kobe_{i+1:02d}.png")
        image.save(path)
        print(f"Saved: {path}")
