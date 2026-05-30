import os
from pathlib import Path

final_dir = Path("/home/user/barber_ai/training_data/final_set")
trigger_word = "abcstyle" # Unique trigger for AI Beauty Concepts

for img_p in final_dir.glob("*.png"):
    caption_p = img_p.with_suffix('.txt')
    with open(caption_p, 'w') as f:
        # High-quality tokens to reinforce realism
        f.write(f"{trigger_word} hair, kinky straight texture, highly detailed skin pores, 8k uhd, soft studio lighting, blurred background top")
    print(f"✅ Captioned: {img_p.name}")

print("\n🚀 Captions ready for the 4090.")
