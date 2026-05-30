import torch
from safetensors.torch import load_file, save_file
from collections import OrderedDict

src  = "/home/kenny/barber_ai/lora_outputs/razor_designs/razor_designs_lora.safetensors"
dest = "/home/kenny/barber_ai/lora_outputs/razor_designs/razor_designs_lora_diffusers.safetensors"

print("Loading LoRA...")
weights = load_file(src)

converted = OrderedDict()
for key, value in weights.items():
    new_key = key
    new_key = new_key.replace("lora_unet_", "unet.")
    new_key = new_key.replace("lora_te1_", "text_encoder.")
    new_key = new_key.replace("lora_te2_", "text_encoder_2.")
    new_key = new_key.replace("_lora_down.weight", ".lora_A.weight")
    new_key = new_key.replace("_lora_up.weight",   ".lora_B.weight")
    converted[new_key] = value

save_file(converted, dest)
print(f"✅ Converted LoRA saved: {dest}")
