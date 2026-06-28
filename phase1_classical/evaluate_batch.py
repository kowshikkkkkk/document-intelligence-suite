"""
Phase 1: Batch evaluation of rule-based extraction accuracy.

Runs the OCR + rule-extraction pipeline across a batch of test receipts,
compares each extracted field against ground truth, and reports accuracy
per field. This gives us real numbers for the Phase 1 row in our final
3-tier comparison table.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "data"))
from extract_rules import extract_fields

# Point this at the TEST split, since that's the standard set for evaluation
# (we trained/tuned our rules by eyeballing train examples, test is unseen)
TEST_ROOT = Path(__file__).parent.parent / "data" / "raw" / "SROIE2019" / "test"
IMG_DIR = TEST_ROOT / "img"
ENTITIES_DIR = TEST_ROOT / "entities"


def load_ground_truth(receipt_id: str) -> dict:
    label_path = ENTITIES_DIR / f"{receipt_id}.txt"
    raw_text = label_path.read_text(encoding="utf-8", errors="ignore")
    return json.loads(raw_text)


def normalize(value) -> str:
    """Lowercase + strip whitespace, so minor formatting diffs don't
    count as a mismatch unfairly."""
    if value is None:
        return ""
    return str(value).strip().lower()


def evaluate_batch(num_samples: int = 20):
    all_ids = sorted(p.stem for p in IMG_DIR.glob("*.jpg"))
    sample_ids = all_ids[:num_samples]

    results = []
    correct_counts = {"company": 0, "date": 0, "total": 0}

    for i, receipt_id in enumerate(sample_ids):
        print(f"[{i+1}/{len(sample_ids)}] Processing {receipt_id}...")

        try:
            predicted = extract_fields(receipt_id, base_dir=TEST_ROOT)
            truth = load_ground_truth(receipt_id)
        except Exception as e:
            print(f"  Skipped due to error: {e}")
            continue

        row = {"id": receipt_id}
        for field in ["company", "date", "total"]:
            pred_val = normalize(predicted.get(field))
            true_val = normalize(truth.get(field))
            is_match = pred_val == true_val and pred_val != ""
            row[field] = {"predicted": predicted.get(field), "truth": truth.get(field), "match": is_match}
            if is_match:
                correct_counts[field] += 1

        results.append(row)

    total = len(results)
    print("\n" + "=" * 50)
    print(f"RESULTS ({total} receipts evaluated)")
    print("=" * 50)
    for field, correct in correct_counts.items():
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"{field:10s}: {correct}/{total} correct  ({accuracy:.1f}%)")

    return results


if __name__ == "__main__":
    evaluate_batch(num_samples=20)