"""
Phase 2, Step 1: Build labeled training data from OCR output + ground truth.

For every training receipt, runs OCR to get text chunks, then labels each
chunk as one of: "company", "date", "total", or "other" by comparing it
against the ground truth fields.

Output: a JSON file with rows of {text, label} pairs, which Phase 2's
PyTorch dataset loader will consume directly.
"""

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "data"))
from run_ocr import run_ocr_on_receipt, DATA_ROOT

TRAIN_ROOT = DATA_ROOT / "train"
ENTITIES_DIR = TRAIN_ROOT / "entities"
OUTPUT_PATH = Path(__file__).parent / "labeled_chunks.json"

COMPANY_SIMILARITY_THRESHOLD = 0.6


def fuzzy_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def label_chunk(chunk_text: str, truth: dict) -> str:
    chunk_clean = chunk_text.strip().lower()

    # Date and total: short, exact-ish values -- check substring containment
    # (handles cases like OCR chunk being "Total 9.00" containing "9.00")
    date_val = str(truth.get("date", "")).strip().lower()
    total_val = str(truth.get("total", "")).strip().lower()

    if date_val and date_val in chunk_clean:
        return "date"
    if total_val and total_val in chunk_clean:
        return "total"

    # Company: longer, messier OCR text -- use fuzzy similarity instead
    company_val = str(truth.get("company", "")).strip().lower()
    if company_val and fuzzy_similarity(chunk_clean, company_val) > COMPANY_SIMILARITY_THRESHOLD:
        return "company"

    return "other"


def load_ground_truth(receipt_id: str) -> dict:
    label_path = ENTITIES_DIR / f"{receipt_id}.txt"
    raw_text = label_path.read_text(encoding="utf-8", errors="ignore")
    return json.loads(raw_text)


def build_dataset(limit: int = None):
    img_dir = TRAIN_ROOT / "img"
    all_ids = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if limit:
        all_ids = all_ids[:limit]

    labeled_rows = []
    label_counts = {"company": 0, "date": 0, "total": 0, "other": 0}

    for i, receipt_id in enumerate(all_ids):
        print(f"[{i+1}/{len(all_ids)}] Processing {receipt_id}...")
        try:
            truth = load_ground_truth(receipt_id)
            ocr_results = run_ocr_on_receipt(receipt_id, base_dir=TRAIN_ROOT)
        except Exception as e:
            print(f"  Skipped due to error: {e}")
            continue

        for _, text, _ in ocr_results:
            if not text.strip():
                continue
            label = label_chunk(text, truth)
            labeled_rows.append({"text": text, "label": label, "receipt_id": receipt_id})
            label_counts[label] += 1

    print("\n" + "=" * 50)
    print(f"Built {len(labeled_rows)} labeled chunks from {len(all_ids)} receipts (before balancing)")
    for label, count in label_counts.items():
        print(f"  {label:10s}: {count}")

    # Undersample "other" so it doesn't completely dominate the dataset.
    # Keep roughly 4x the count of our largest minority class, as a
    # reasonable middle ground (not perfectly balanced, but not 95% either).
    import random
    random.seed(42)  # reproducible shuffling

    other_rows = [r for r in labeled_rows if r["label"] == "other"]
    keep_rows = [r for r in labeled_rows if r["label"] != "other"]

    max_minority_count = max(label_counts["company"], label_counts["date"], label_counts["total"])
    other_keep_count = min(len(other_rows), max_minority_count * 4)

    random.shuffle(other_rows)
    balanced_rows = keep_rows + other_rows[:other_keep_count]
    random.shuffle(balanced_rows)

    final_counts = {"company": 0, "date": 0, "total": 0, "other": 0}
    for r in balanced_rows:
        final_counts[r["label"]] += 1

    print(f"\nAfter undersampling 'other' (kept {other_keep_count} of {len(other_rows)}):")
    for label, count in final_counts.items():
        print(f"  {label:10s}: {count}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(balanced_rows, f, indent=2)
    print(f"\nSaved {len(balanced_rows)} balanced rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    # Start small to verify this works before running on all 626 receipts
    build_dataset(limit=150)