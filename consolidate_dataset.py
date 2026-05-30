"""
Master Dataset Consolidator
============================
Merges all training images + captions into one clean deduplicated dataset.
Renames files sequentially, copies matched PNG+TXT pairs only.

Run:
    python3 consolidate_dataset.py
"""

import os
import shutil
import hashlib
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────────
SOURCE_DIRS = [
    "/home/kenny/barber_ai/training_data/lora",
    "/home/kenny/barber_ai/training_data/razor_designs_lora/10_razor",
    "/home/kenny/barber_ai/training_data/lora/100_haircut_design_v1",
    "/home/kenny/barber_ai/training_data/final_designs",
]

OUTPUT_DIR = "/home/kenny/barber_ai/training_data/master_dataset"
# ────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

print("=" * 55)
print("  Master Dataset Consolidator")
print("=" * 55)

seen_hashes = {}
pairs = []
skipped_no_caption = []
skipped_duplicate = []

# ── Collect all PNG+TXT pairs ───────────────────────────────────
for src_dir in SOURCE_DIRS:
    src = Path(src_dir)
    if not src.exists():
        print(f"  WARNING: folder not found — {src_dir}")
        continue

    pngs = sorted(src.glob("*.png"))
    print(f"\n  Scanning: {src_dir}")
    print(f"  Found:    {len(pngs)} PNG files")

    for png in pngs:
        txt = png.with_suffix(".txt")

        # Must have matching caption
        if not txt.exists():
            skipped_no_caption.append(png.name)
            continue

        # Deduplicate by image hash
        h = get_hash(png)
        if h in seen_hashes:
            skipped_duplicate.append(png.name)
            continue

        seen_hashes[h] = png
        pairs.append((png, txt))

# ── Copy to master dataset with sequential naming ───────────────
print(f"\n  Copying {len(pairs)} unique pairs → {OUTPUT_DIR}")

for idx, (png, txt) in enumerate(tqdm(pairs, desc="  Copying")):
    new_name = f"master_{idx:04d}"
    dest_png = Path(OUTPUT_DIR) / f"{new_name}.png"
    dest_txt = Path(OUTPUT_DIR) / f"{new_name}.txt"

    shutil.copy2(png, dest_png)
    shutil.copy2(txt, dest_txt)

# ── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Consolidation Complete")
print("=" * 55)
print(f"  Total pairs copied   : {len(pairs)}")
print(f"  Duplicates removed   : {len(skipped_duplicate)}")
print(f"  Skipped (no caption) : {len(skipped_no_caption)}")
print(f"\n  Master dataset ready : {OUTPUT_DIR}")

if pairs:
    print(f"  Files: master_0000.png/txt → master_{len(pairs)-1:04d}.png/txt")

if skipped_no_caption:
    print(f"\n  Images missing captions:")
    for f in skipped_no_caption[:10]:
        print(f"    {f}")
    if len(skipped_no_caption) > 10:
        print(f"    ... and {len(skipped_no_caption)-10} more")

print("\n  Next step — update your training config to:")
print(f"    dataset_dir = {OUTPUT_DIR}")
print("=" * 55)
