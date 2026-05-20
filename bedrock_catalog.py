import boto3
import json
import base64
import os
import time

client = boto3.client('bedrock-runtime', region_name='us-west-2')

STYLES = {
    "barbershop": [
        ("bald_fade", "clean bald fade haircut, skin fade sides, Black man, professional barbershop photo, realistic, studio lighting, barber cape"),
        ("low_fade", "low fade haircut, natural hair on top, Black man, professional barbershop photo, realistic, studio lighting, barber cape"),
        ("mid_fade", "mid fade haircut, waves on top, Black man, professional barbershop photo, realistic, studio lighting, barber cape"),
        ("high_fade", "high fade haircut, short hair on top, Black man, professional barbershop photo, realistic, studio lighting, barber cape"),
        ("taper_fade", "taper fade haircut, clean neckline, Black man, professional barbershop photo, realistic, studio lighting, barber cape"),
        ("saints_fade", "low fade with hard part, natural curly hair on top, Black man, professional barbershop photo, realistic"),
        ("lineup", "sharp lineup edge up, natural hair, Black man, professional barbershop photo, realistic, studio lighting"),
        ("360_waves", "360 waves haircut, low fade, Black man, professional barbershop photo, realistic, studio lighting"),
        ("caesar_cut", "caesar cut haircut, low fade, Black man, professional barbershop photo, realistic, studio lighting"),
        ("temp_fade", "temple fade haircut, natural hair on top, Black man, professional barbershop photo, realistic"),
    ],
    "braids": [
        ("box_braids", "box braids hairstyle, Black man, professional photo, realistic, studio lighting"),
        ("cornrows", "cornrows hairstyle, neat rows, Black man, professional photo, realistic, studio lighting"),
        ("feed_in_braids", "feed in braids hairstyle, Black man, professional photo, realistic, studio lighting"),
        ("locs", "starter locs hairstyle, Black man, professional photo, realistic, studio lighting"),
        ("twists", "two strand twists hairstyle, Black man, professional photo, realistic, studio lighting"),
    ],
    "afros": [
        ("afro", "natural afro hairstyle, Black man, professional photo, realistic, studio lighting"),
        ("afro_fade", "afro with fade sides, Black man, professional barbershop photo, realistic, studio lighting"),
        ("coils", "natural coils hairstyle, defined curls, Black man, professional photo, realistic"),
        ("frohawk", "frohawk hairstyle, fade sides, Black man, professional barbershop photo, realistic"),
    ],
    "beards": [
        ("full_beard", "full beard groomed, Black man, professional barbershop photo, realistic, studio lighting"),
        ("goatee", "goatee beard style, Black man, professional barbershop photo, realistic, studio lighting"),
        ("stubble", "clean stubble beard, Black man, professional barbershop photo, realistic, studio lighting"),
        ("beard_fade", "beard fade, shaped beard, Black man, professional barbershop photo, realistic"),
        ("beard_lineup", "beard lineup, sharp edges, Black man, professional barbershop photo, realistic"),
    ],
    "women": [
        ("bob_cut", "bob cut hairstyle, Black woman, professional salon photo, realistic, studio lighting"),
        ("natural_curls", "natural curls hairstyle, Black woman, professional salon photo, realistic"),
        ("protective_style", "protective style braids, Black woman, professional salon photo, realistic"),
        ("silk_press", "silk press hairstyle, Black woman, professional salon photo, realistic, studio lighting"),
        ("goddess_braids", "goddess braids hairstyle, Black woman, professional salon photo, realistic"),
    ],
}

OUTPUT_DIR = "/home/kenny/barber_ai/catalog"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_style(category, style_name, prompt):
    output_path = f"{OUTPUT_DIR}/{category}_{style_name}.jpg"
    if os.path.exists(output_path):
        print(f"Skipping {style_name} - already exists")
        return output_path
    
    try:
        body = json.dumps({
            "prompt": prompt,
            "mode": "text-to-image",
            "output_format": "jpeg",
            "aspect_ratio": "1:1"
        })
        response = client.invoke_model(
            modelId="stability.sd3-5-large-v1:0",
            body=body
        )
        result = json.loads(response['body'].read())
        image_data = base64.b64decode(result['images'][0])
        with open(output_path, 'wb') as f:
            f.write(image_data)
        print(f"✅ Generated: {category}/{style_name}")
        time.sleep(1)
        return output_path
    except Exception as e:
        print(f"❌ Failed {style_name}: {e}")
        return None

if __name__ == "__main__":
    print("Starting catalog generation...")
    total = sum(len(v) for v in STYLES.values())
    done = 0
    for category, styles in STYLES.items():
        os.makedirs(f"{OUTPUT_DIR}/{category}", exist_ok=True)
        for style_name, prompt in styles:
            generate_style(category, style_name, prompt)
            done += 1
            print(f"Progress: {done}/{total}")
    print("Catalog generation complete!")
    print(f"Images saved to: {OUTPUT_DIR}")
