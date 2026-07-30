# ==========================================================
# TEST OCR FOR ALL PDF PAGES
# ==========================================================

from pathlib import Path
from ocr_service import extract_text

print("-------------------------------------")
print("Starting OCR Test")
print("-------------------------------------")

# Folder jahan PDF ki images bani hain
OUTPUT_FOLDER = Path("output")

# OCR result save karne ki file
RESULT_FILE = OUTPUT_FOLDER / "ocr_result.txt"

# Saare page_*.png files uthao
images = sorted(OUTPUT_FOLDER.glob("page_*.png"))

if not images:
    print("No page images found!")
    exit()

all_text = []

for i, image in enumerate(images, start=1):

    print(f"\nOCR on {image.name}")

    text = extract_text(str(image))

    # Terminal mein bhi dikhao
    print(text)

    # File ke liye store karo
    all_text.append(f"\n\n========== PAGE {i} ==========\n\n")
    all_text.append(text)

# Sab pages ka combined text file mein save karo
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    f.write("".join(all_text))

print("\n===================================")
print("OCR Finished Successfully")
print("===================================")

print(f"\nSaved OCR Result at:\n{RESULT_FILE}")