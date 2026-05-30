import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image

def generate_final_luxury(prompt, output_name):
    model_id = "runwayml/stable-diffusion-v1-5"
    lora_path = "/home/user/barber_ai/outputs/hair_beauty_v1.safetensors"
    
    # 1. Base Generation
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.load_lora_weights(lora_path, adapter_name="hair")
    
    print("🎨 Sculpting the silhouette...")
    # We use (hairbeauty:1.2) syntax to boost the specific LoRA keyword
    base_img = pipe(
        prompt=f"a high-end studio portrait of a beautiful woman, {prompt}, (hairbeauty:1.2), highly detailed kinky coily hair texture, luxury fashion photography",
        negative_prompt="multiple heads, extra limbs, cartoon, smooth plastic hair, blurry, low quality",
        num_inference_steps=30,
        width=512,
        height=512,
        cross_attention_kwargs={"scale": 0.75} # Global LoRA strength
    ).images[0]

    # 2. Detail Refinement (The 'Texture Pass')
    print("🔍 Sharpening the taper fade...")
    upscale_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    upscale_pipe.load_lora_weights(lora_path, adapter_name="hair")
    
    base_img_large = base_img.resize((1024, 1024), Image.LANCZOS)
    
    # We crank the scale to 0.95 ONLY for the texture pass
    final_img = upscale_pipe(
        prompt=f"8k UHD, extremely detailed (coily kinky hair texture:1.3), sharp taper fade, hairbeauty",
        image=base_img_large,
        strength=0.4, # Keep the woman, change the hair detail
        num_inference_steps=25,
        cross_attention_kwargs={"scale": 0.95} 
    ).images[0]
    
    final_img.save(f"/home/user/barber_ai/outputs/{output_name}.png")
    print(f"✅ FINAL Luxury result saved as {output_name}.png")

if __name__ == "__main__":
    generate_final_luxury("a woman with a tight taper fade", "hair_beauty_final_v4")
