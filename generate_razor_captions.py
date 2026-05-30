import os
import random
from pathlib import Path

IMAGE_DIR = Path("/home/kenny/barber_ai/resized_images")

captions = [
    # BASE RAZOR ART STYLE
    "RazorArtStyle, photorealistic hair engraving, detailed portrait shaved into skin fade, dark hair stubble creates shading, lighter stubble creates highlights, black background, barber cape, side of head",
    "RazorArtStyle, hair tattoo art, geometric design engraved into fade, dark stubble gradient shading, photorealistic barbershop engraving, black background, back of head, side of head",
    "RazorArtStyle, precision hair engraving, clipper art shaved into fade, stubble depth creates image contrast, professional barber art, dark background, side profile",
    "RazorArtStyle, fade hair engraving, detailed design cut into dark hair, skin fade gradient, photorealistic hair art, black background, barber cape visible",

    # PORTRAIT ENGRAVINGS (KOBE, ATHLETES, CELEBRITIES)
    "RazorArtStyle, portrait hair engraving, realistic face carved into fade, stubble shading builds facial structure, deep razor lines define features, black background, side of head",
    "RazorArtStyle, athlete portrait shaved into fade, high-contrast stubble shading, precise clipper engraving, photorealistic hair tattoo, dark background, side profile",
    "RazorArtStyle, celebrity portrait hair engraving, detailed facial features carved with razor lines, smooth fade gradient, black background, barber cape",
    "RazorArtStyle, iconic portrait hair tattoo, deep stubble shadows, light stubble highlights, precision barber art, back of head, dark background",

    # SPORTS LOGOS
    "RazorArtStyle, sports logo hair engraving, bold razor lines carve team emblem into fade, high-contrast stubble shading, black background, side of head",
    "RazorArtStyle, basketball logo shaved into fade, circular outline with deep razor cuts, stubble gradient shading, photorealistic barber art, dark background",
    "RazorArtStyle, football team logo hair tattoo, sharp geometric razor lines, stubble depth creates logo contrast, black background, back of head",
    "RazorArtStyle, baseball logo hair engraving, curved razor lines, stubble shading for highlights, photorealistic fade art, dark background",

    # ADVANCED DETAIL / DEPTH / SHADING
    "RazorArtStyle, ultra-detailed hair engraving, micro-stubble shading for depth, razor-sharp linework, photorealistic fade art, black background",
    "RazorArtStyle, high-contrast hair tattoo, deep stubble shadows, crisp razor outlines, professional barber engraving, side profile",
    "RazorArtStyle, 3D hair engraving effect, layered stubble shading, precision razor carving, dark background, back of head",

    # ANGLES / VIEWS
    "RazorArtStyle, hair engraving close-up, macro detail of stubble shading and razor lines, black background",
    "RazorArtStyle, hair tattoo from back angle, full design visible, fade gradient, photorealistic barber art",
    "RazorArtStyle, side profile hair engraving, portrait carved into fade, stubble shading, dark background",
]

for img in IMAGE_DIR.glob("*.*"):
    if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        continue

    txt_path = img.with_suffix(".txt")
    if txt_path.exists():
        continue

    caption = random.choice(captions)
    txt_path.write_text(caption)

print("✅ Captions generated for all images.")
