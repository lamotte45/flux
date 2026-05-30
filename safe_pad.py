import cv2
import numpy as np
from pathlib import Path

raw_dir = Path("/home/user/barber_ai/raw_photos")
out_dir = Path("/home/user/barber_ai/training_data/final_set")
out_dir.mkdir(parents=True, exist_ok=True)

for img_p in raw_dir.glob("*.[jp][pn]g"):
    img = cv2.imread(str(img_p))
    h, w = img.shape[:2]
    
    # Add 150px of blurred padding to the top
    top_pad = 150
    # Create a blurred version of the top of the image to use as padding
    top_slice = img[0:10, :, :]
    padding = cv2.resize(top_slice, (w, top_pad), interpolation=cv2.INTER_CUBIC)
    padding = cv2.GaussianBlur(padding, (51, 51), 0)
    
    # Stack them
    combined = np.vstack((padding, img))
    
    # Resize back to 1024x1024
    final = cv2.resize(combined, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
    
    cv2.imwrite(str(out_dir / img_p.name), final)
    print(f"✅ Safe Padded: {img_p.name}")

print("\n🚀 Pure, high-res dataset ready. No AI artifacts.")
