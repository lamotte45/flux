import os
import random

DATASET_DIR = "/home/kenny/barber_ai/training_data/lora"

BAD_WORDS = [
    "logo", "sketch", "minimal", "engraved", "graphic", "vector"
]

VIEWS = [
    "back of head view",
    "side profile view",
    "rear angle view",
    "close-up haircut view"
]

def clean_caption(text):
    text = text.lower()

    # Remove bad words
    for word in BAD_WORDS:
        text = text.replace(word, "")

    view = random.choice(VIEWS)

    return (
        f"{view}, african american male, "
        "deep shaved hair design carved into fade, "
        "bold razor carved lines, high contrast hair carving, "
        "clean fade haircut, realistic hair texture"
    )

def process_dataset():
    count = 0

    for file in os.listdir(DATASET_DIR):
        if file.endswith(".txt"):
            path = os.path.join(DATASET_DIR, file)

            with open(path, "r") as f:
                original = f.read()

            fixed = clean_caption(original)

            with open(path, "w") as f:
                f.write(fixed)

            count += 1

    print(f"🔥 Fixed {count} captions with mixed angles!")

if __name__ == "__main__":
    process_dataset()
