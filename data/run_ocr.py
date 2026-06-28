"""
Phase 1, Step 1: Run OCR on a single receipt image.

This script loads one receipt image and runs EasyOCR on it to extract
raw text. The goal here is just to SEE what OCR output looks like before
we try to write any rules to pull out company/date/total from it.

EasyOCR returns a list of (bounding_box, text, confidence) for every
chunk of text it detects in the image -- not a clean paragraph.
"""

import easyocr
from pathlib import Path

# Anchor this path to the location of this script file itself,
# so it works correctly no matter which folder you run it from.
IMG_DIR = Path(__file__).parent / "raw" / "SROIE2019" / "train" / "img"


DATA_ROOT = Path(__file__).parent / "raw" / "SROIE2019"


def run_ocr_on_receipt(receipt_id: str, base_dir: Path = None):
    """
    base_dir: the split folder to use, e.g. DATA_ROOT / "train" or
    DATA_ROOT / "test". Defaults to train if not specified, to keep
    older calls working without changes.
    """
    if base_dir is None:
        base_dir = DATA_ROOT / "train"

    img_path = base_dir / "img" / f"{receipt_id}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"No image found at {img_path}")

    # 'en' = English. gpu=False forces CPU since your GPU has limited
    # VRAM and OCR doesn't need much speed for a single image like this.
    reader = easyocr.Reader(["en"], gpu=False)

    # detail=1 gives us (bounding_box, text, confidence) per detected chunk
    results = reader.readtext(str(img_path), detail=1)

    return results


if __name__ == "__main__":
    test_id = "X00016469612"  # same receipt we looked at before
    print(f"Running OCR on {test_id}...\n")

    results = run_ocr_on_receipt(test_id)

    print(f"Found {len(results)} text regions:\n")
    for i, (bbox, text, confidence) in enumerate(results):
        print(f"{i:2d}. '{text}'  (confidence: {confidence:.2f})")