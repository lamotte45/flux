import os
from PIL import Image

# Setup folders
input_dir = "input_images"
output_dir = "resized_images"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"🚀 Resizing images from {input_dir}...")

count = 0
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            with Image.open(os.path.join(input_dir, filename)) as img:
                img = img.convert("RGB")
                img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
                img.save(os.path.join(output_dir, filename))
                count += 1
                if count % 20 == 0:
                    print(f"✅ Processed {count} images...")
        except Exception as e:
            print(f"❌ Failed {filename}: {e}")

print(f"🏁 Finished! {count} images resized in '{output_dir}'")
