import os
import shutil
from pathlib import Path

SOURCE_DIR = "/home/kenny/barber_ai/training_data"
OUTPUT_DIR = "/home/kenny/barber_ai/training_data/micro_geometric"

os.makedirs(OUTPUT_DIR, exist_ok=True)

KEYWORDS = [
    "design",
    "pattern",
    "geometric",
    "razor",
    "basketball",
    "lines",
    "carved"
]

copied = 0

for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.endswith(".txt"):
            txt_path = os.path.join(root, file)

            try:
                with open(txt_path, "r", errors="ignore") as f:
                    text = f.read().lower()
            except:
                continue

            if any(k in text for k in KEYWORDS):
                img_path = txt_path.replace(".txt", ".png")

                if os.path.exists(img_path):
                    new_img = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
                    new_txt = os.path.join(OUTPUT_DIR, os.path.basename(txt_path))

                    shutil.copy2(img_path, new_img)
                    shutil.copy2(txt_path, new_txt)

                    print(f"✅ Copied: {file}")
                    copied += 1

print(f"\n🔥 DONE — {copied} design images extracted")
