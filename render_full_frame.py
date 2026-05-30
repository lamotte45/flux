import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("models/hair_beauty_final.safetensors", adapter_name="hair")
pipe.set_adapters(["hair"], adapter_weights=[0.7])

# Added "cinematic wide shot" and "upper body" to force a zoom out
prompt = (
    "a cinematic wide shot, upper body photo of hairbeauty, "
    "showing the full head of hair and top of head, voluminous kinky straight hair, "
    "natural black color, realistic skin pores, high quality studio lighting, 8k"
)

negative_prompt = "close up, cropped head, chopped off hair, top of head missing, face only, blurry, plastic"

print("🎨 Rendering with wide framing to see the full head...")
with torch.inference_mode():
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=50, 
        guidance_scale=7.0,
        height=896, # Extra vertical space
        width=512
    ).images[0]

image.save("outputs/full_frame_result.png")
print("✅ SUCCESS: Full frame image saved to outputs/full_frame_result.png")
