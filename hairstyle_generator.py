import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline, 
    ControlNetModel, 
    AutoencoderKL,
    EulerDiscreteScheduler
)

# 1. SETUP PIPELINE (Optimized for 4090)
device = "cuda"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_id = "diffusers/controlnet-canny-sdxl-1.0"
vae_id = "madebyollin/sdxl-vae-fp16-fix"

print("🎨 Loading AI Hair Engine into VRAM...")

# Load ControlNet
controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16).to(device)
# Load VAE (Prevents "black image" bugs in SDXL)
vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16).to(device)

# Build Pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to(device)

# Load your custom "Platinum Waves" LoRA
# Ensure this path matches where you saved your LoRA .safetensors file
LORA_PATH = "/home/user/barber_ai/models/lora_v3/platinum_waves_ultra_3D.safetensors"
pipe.load_lora_weights(LORA_PATH)

# Use a fast scheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

# Memory Optimization
pipe.enable_model_cpu_offload() # Saves VRAM for the Emotion Engine
pipe.enable_xformers_memory_efficient_attention()

def generate_style_variants(selfie_path, style_prompt, count=6):
    """
    Takes a selfie, locks the structure with Canny, 
    and generates 6 variants using the LoRA.
    """
    # Load and prep image
    init_image = Image.open(selfie_path).convert("RGB").resize((1024, 1024))
    image_np = np.array(init_image)
    
    # Run Canny Edge Detection (Structure Lock)
    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(image_np, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    canny_image = Image.fromarray(edges)

    # Master Prompt (Triggering your LoRA)
    # Note: Replace 'platinum_waves_trigger' with your actual trained keyword
    full_prompt = f"{style_prompt}, platinum_waves_trigger, highly detailed hair, professional salon lighting, 8k"
    negative_prompt = "bald, blurry, distorted face, bad anatomy, low quality, orange hair"

    generated_paths = []

    print(f"⚡ Generating {count} variants for: {style_prompt}...")
    
    for i in range(count):
        # Generate with random seed for variety
        generator = torch.Generator(device=device).manual_seed(np.random.randint(0, 10**6))
        
        output = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            controlnet_conditioning_scale=0.6, # 0.6 keeps the face/head but allows hair change
            cross_attention_kwargs={"scale": 0.8}, # LoRA strength
            num_inference_steps=25,
            generator=generator
        ).images[0]

        # Save result
        save_path = f"results/style_var_{i}_{int(time.time())}.png"
        output.save(save_path)
        generated_paths.append(save_path)

    return generated_paths

if __name__ == "__main__":
    # Test run
    print("🧪 Testing real hair generation...")
    import time
    test_paths = generate_style_variants("uploads/test_selfie.png", "Platinum Waves", count=1)
    print(f"✅ Test complete. Created: {test_paths}")
