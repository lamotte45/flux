import os
from PIL import Image
from pathlib import Path

def prepare_dataset(input_dir, output_dir, size=(1024, 1024), trigger_word="hairbeauty"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # This creates the '10_hairbeauty' folder Kohya needs
    output_path.mkdir(parents=True, exist_ok=True)

    supported_formats = (".png", ".jpg", ".jpeg", ".webp")
    count = 0

    print(f"🔍 Searching for images in {input_path}...")

    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in supported_formats:
            try:
                with Image.open(img_file) as img:
                    img = img.convert("RGB")
                    
                    # Smart Center Crop
                    width, height = img.size
                    min_dim = min(width, height)
                    left = (width - min_dim) / 2
                    top = (height - min_dim) / 2
                    right = (width + min_dim) / 2
                    bottom = (height + min_dim) / 2
                    
                    img = img.crop((left, top, right, bottom))
                    img = img.resize(size, Image.LANCZOS)

                    base_name = f"style_{count:03d}"
                    img.save(output_path / f"{base_name}.png", "PNG")

                    # Create the caption file
                    with open(output_path / f"{base_name}.txt", "w") as f:
                        f.write(f"a professional photo of {trigger_word}, high quality, realistic texture")

                    count += 1
            except Exception as e:
                print(f"Skipping {img_file}: {e}")

    print(f"✅ SUCCESS: Processed {count} images into {output_dir}")

if __name__ == "__main__":
    prepare_dataset("/home/user/barber_ai/raw_photos", "/home/user/barber_ai/training_data/10_hairbeauty")
