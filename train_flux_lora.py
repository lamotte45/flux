import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from diffusers import FluxPipeline
from diffusers.loaders import AttnProcsLayers
from PIL import Image
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "black-forest-labs/FLUX.1-schnell"
DATA_DIR = "/home/kenny/barber_ai/training_data/lora"
OUTPUT_DIR = "/home/kenny/barber_ai/models/lora_flux"
BATCH_SIZE = 1
LR = 1e-4
MAX_STEPS = 1500
RANK = 16
RES = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# DATASET
# -----------------------------
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.paths = [str(p) for p in Path(folder).glob("*.jpg")]
        self.transform = transforms.Compose([
            transforms.Resize((RES, RES)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), "hair design razor fade"

dataset = ImageFolderDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# ACCELERATOR
# -----------------------------
accelerator = Accelerator(mixed_precision="bf16")

# -----------------------------
# LOAD MODEL
# -----------------------------
pipe = FluxPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)

transformer = pipe.transformer  # FLUX uses a transformer, not a UNet

# Inject LoRA into transformer attention processors
lora_attn_procs = {}
for name, module in transformer.attn_processors.items():
    lora_attn_procs[name] = AttnProcsLayers(
        hidden_size=module.hidden_size,
        cross_attention_dim=module.cross_attention_dim,
        rank=RANK
    )

transformer.set_attn_processor(lora_attn_procs)

optimizer = torch.optim.AdamW(transformer.parameters(), lr=LR)

transformer, optimizer, loader = accelerator.prepare(transformer, optimizer, loader)

# -----------------------------
# TRAIN LOOP
# -----------------------------
step = 0
transformer.train()

for epoch in range(9999):
    for batch, captions in loader:
        step += 1
        if step > MAX_STEPS:
            break

        images = batch.to(accelerator.device, dtype=torch.bfloat16)

        noise = torch.randn_like(images)
        noisy = images + noise * 0.1

        pred = transformer(noisy, timestep=10, encoder_hidden_states=None).sample

        loss = torch.nn.functional.mse_loss(pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            accelerator.print(f"Step {step} | Loss {loss.item():.4f}")

    if step > MAX_STEPS:
        break

# -----------------------------
# SAVE LORA
# -----------------------------
accelerator.wait_for_everyone()

if accelerator.is_main_process:
    transformer.save_attn_procs(OUTPUT_DIR)
    print(f"Saved LoRA to {OUTPUT_DIR}")

