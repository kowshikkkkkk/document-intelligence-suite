"""
Phase 0: Load and inspect a single paired receipt sample.

Given a receipt ID (e.g. "X00016469612"), this script:
1. Builds the path to the matching image file (train/img/<id>.jpg)
2. Builds the path to the matching entity/label file (train/entities/<id>.txt)
3. Loads and parses the label JSON
4. Prints the label fields
5. Opens the image in your default image viewer so you can visually check it
   against the printed label

This pairing logic (id -> image path + label path) is the foundation for
Phase 1 (rules baseline), Phase 2 (custom PyTorch dataset loader), and
Phase 3 (VLM fine-tuning dataset prep) -- so it's worth getting comfortable
with this small script before moving on.
"""

import json
from pathlib import Path
from PIL import Image

# Root of the extracted dataset. Adjust if your folder name differs.
DATA_ROOT = Path("raw/SROIE2019/train")
IMG_DIR = DATA_ROOT / "img"
ENTITIES_DIR = DATA_ROOT / "entities"


def load_sample(receipt_id: str):
    """
    Given a receipt ID like 'X00016469612', return (image, label_dict).
    Raises FileNotFoundError with a clear message if either file is missing.
    """
    img_path = IMG_DIR / f"{receipt_id}.jpg"
    label_path = ENTITIES_DIR / f"{receipt_id}.txt"

    if not img_path.exists():
        raise FileNotFoundError(f"No image found at {img_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"No label file found at {label_path}")

    image = Image.open(img_path)

    # The entity files are JSON, but some SROIE entity files have minor
    # formatting quirks, so we read raw text first and parse it ourselves
    # rather than assuming it's always perfectly clean JSON.
    raw_text = label_path.read_text(encoding="utf-8", errors="ignore")
    label = json.loads(raw_text)

    return image, label


def list_available_ids(limit: int = 5):
    """Helper: list a few receipt IDs actually present in the dataset."""
    ids = [p.stem for p in sorted(IMG_DIR.glob("*.jpg"))]
    return ids[:limit]


if __name__ == "__main__":
    # Quick sanity check: show a few real IDs we can try
    sample_ids = list_available_ids()
    print(f"Example receipt IDs found: {sample_ids}\n")

    # Use the first available ID as our test case
    test_id = sample_ids[0]
    print(f"Loading sample: {test_id}")

    image, label = load_sample(test_id)

    print("\n--- Ground truth label ---")
    for key, value in label.items():
        print(f"{key:10s}: {value}")

    print(f"\nImage size: {image.size} (width x height)")
    print("Opening image now...")
    image.show()

