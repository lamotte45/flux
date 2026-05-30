from PIL import Image, ImageEnhance
import numpy as np
import os
from tsr.utils import remove_background

def prep_for_triposr(path_in, path_out="prepped.png"):
    if not os.path.exists(path_in):
        print(f"❌ Error: {path_in} not found.")
        return None

    print(f"📸 Opening {path_in}...")
    img = Image.open(path_in).convert("RGBA")

    # 1. Contrast Boost
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.4)

    # 2. Remove background
    img_no_bg = remove_background(img)

    # 3. Create a clean white canvas
    bg = Image.new("RGB", img_no_bg.size, (255, 255, 255))
    bg.paste(img_no_bg, mask=img_no_bg.split()[-1])

    # 4. TIGHT BARBER CROP (Removes shoulders/pancake)
    print("📐 Running Barber-Crop (Head Only)...")
    arr = np.array(bg)
    mask = (arr[:, :, 0] < 250) | (arr[:, :, 1] < 250) | (arr[:, :, 2] < 250)
    coords = np.argwhere(mask)
    
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        # Calculate height and cut off the bottom 40% (the shoulders/shirt)
        subject_height = y1 - y0
        new_y1 = y0 + int(subject_height * 0.60) 
        
        bg = bg.crop((x0, y0, x1, new_y1))

    # 5. Final Resize
    final_img = bg.resize((512, 512), Image.Resampling.LANCZOS)
    final_img.save(path_out)
    print(f"✅ Saved head-only image as: {path_out}")
    return path_out

if __name__ == "__main__":
    prep_for_triposr("finaltest.png")
