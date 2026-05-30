import os
import random

DATASET_DIR = "/home/kenny/barber_ai/training_data/final_designs"

BAD_WORDS = [
    "logo", "sketch", "minimal", "engraved", "graphic", "vector"
]

VIEWS = [
    "back of head view",
    "side profile view",
    "rear angle view",
    "close-up haircut view"
]

DESIGN_TYPES = {
    "basketball": "circular shaved design with curved intersecting razor lines forming basketball seams",
    "soccer": "hexagonal and pentagonal shaved pattern forming soccer ball design",
    "letter": "large bold letter shaved design carved into fade with sharp razor edges",
    "abstract": "geometric shaved design with sharp angular razor lines and patterns"
}

def detect_design_type(filename):
    name = filename.lower()
    if "basketball" in name:
        return "basketball"
    elif "soccer" in name:
        return "soccer"
    elif "letter" in name or "number" in name:
        return "letter"
    else:
        return "abstract"

def clean_caption(file):
    view = random.choice(VIEWS)
    design_type = detect_design_type(file)
    design_desc = DESIGN_TYPES[design_type]

    return (
        f"{view}, african american male haircut, "
        f"deep shaved hair design carved into fade, "
        f"{design_desc}, "
        "bold razor carved lines, high contrast hair carving, "
        "clean fade haircut, realistic hair texture"
    )

def process_dataset():
    count = 0

    for file in os.listdir(DATASET_DIR):
        if file.endswith(".txt"):
            path = os.path.join(DATASET_DIR, file)

            fixed = clean_caption(file)

            with open(path, "w") as f:
                f.write(fixed)

            count += 1

    print(f"🔥 Fixed {count} captions in FINAL dataset!")

if __name__ == "__main__":
    process_dataset()
