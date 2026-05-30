import os

DATASET = "/home/kenny/barber_ai/training_data/master_dataset"

files = os.listdir(DATASET)

png = [f for f in files if f.endswith(".png")]
txt = [f for f in files if f.endswith(".txt")]

print("=" * 50)
print("DATASET CHECK")
print("=" * 50)

print(f"PNG files : {len(png)}")
print(f"TXT files : {len(txt)}")

# Check matching pairs
missing_txt = []
missing_png = []

for p in png:
    if p.replace(".png", ".txt") not in txt:
        missing_txt.append(p)

for t in txt:
    if t.replace(".txt", ".png") not in png:
        missing_png.append(t)

print("\nMissing TXT for PNG:", len(missing_txt))
print("Missing PNG for TXT:", len(missing_png))

if missing_txt[:5]:
    print("\nExamples missing TXT:")
    for f in missing_txt[:5]:
        print(" ", f)

if missing_png[:5]:
    print("\nExamples missing PNG:")
    for f in missing_png[:5]:
        print(" ", f)

print("\nDataset path:", DATASET)
print("=" * 50)
