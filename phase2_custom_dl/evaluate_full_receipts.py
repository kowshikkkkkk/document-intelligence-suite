"""
Phase 2, Step 4: Evaluate the trained classifier on full test receipts.

For each test receipt:
  1. Run OCR to get text chunks
  2. Embed each chunk with the same sentence-transformer used in training
  3. Classify each chunk with our trained model
  4. For each field (company/date/total), pick the chunk with the highest
     confidence for that class
  5. Compare against ground truth (same matching logic as Phase 1, for a
     fair head-to-head comparison)
"""

import json
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent / "data"))
from run_ocr import run_ocr_on_receipt, DATA_ROOT

from train_classifier import ChunkClassifier, LABEL_NAMES

TEST_ROOT = DATA_ROOT / "test"
IMG_DIR = TEST_ROOT / "img"
ENTITIES_DIR = TEST_ROOT / "entities"
MODEL_PATH = Path(__file__).parent / "classifier.pt"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_ground_truth(receipt_id: str) -> dict:
    label_path = ENTITIES_DIR / f"{receipt_id}.txt"
    raw_text = label_path.read_text(encoding="utf-8", errors="ignore")
    return json.loads(raw_text)


def normalize(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def is_match(predicted: str, truth: str) -> bool:
    """Same logic style as Phase 1: substring containment for short
    fields like date/total. Good enough since company name matching
    is handled by the classifier itself now, not string heuristics."""
    pred_norm = normalize(predicted)
    truth_norm = normalize(truth)
    if not pred_norm or not truth_norm:
        return False
    return truth_norm in pred_norm or pred_norm in truth_norm


def predict_fields_for_receipt(receipt_id: str, embed_model, classifier):
    ocr_results = run_ocr_on_receipt(receipt_id, base_dir=TEST_ROOT)
    texts = [text for _, text, _ in ocr_results if text.strip()]

    if not texts:
        return {"company": None, "date": None, "total": None}

    embeddings = embed_model.encode(texts)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    with torch.no_grad():
        logits = classifier(embeddings_tensor)
        probs = F.softmax(logits, dim=1)  # shape: (num_chunks, 4)

    predicted_fields = {}
    for field_name in ["company", "date", "total"]:
        class_idx = LABEL_NAMES.index(field_name)
        confidences_for_field = probs[:, class_idx]
        best_chunk_idx = torch.argmax(confidences_for_field).item()
        predicted_fields[field_name] = texts[best_chunk_idx]

    return predicted_fields


def evaluate_batch(num_samples: int = 20):
    print("Loading embedding model and classifier...")
    embed_model = SentenceTransformer(MODEL_NAME)

    classifier = ChunkClassifier(input_dim=384, hidden_dim=128, num_classes=4)
    classifier.load_state_dict(torch.load(MODEL_PATH))
    classifier.eval()

    all_ids = sorted(p.stem for p in IMG_DIR.glob("*.jpg"))
    sample_ids = all_ids[:num_samples]

    correct_counts = {"company": 0, "date": 0, "total": 0}
    total_evaluated = 0

    for i, receipt_id in enumerate(sample_ids):
        print(f"[{i+1}/{len(sample_ids)}] Processing {receipt_id}...")
        try:
            truth = load_ground_truth(receipt_id)
            predicted = predict_fields_for_receipt(receipt_id, embed_model, classifier)
        except Exception as e:
            print(f"  Skipped due to error: {e}")
            continue

        for field in ["company", "date", "total"]:
            if is_match(predicted.get(field), truth.get(field)):
                correct_counts[field] += 1

        total_evaluated += 1

    print("\n" + "=" * 50)
    print(f"PHASE 2 RESULTS ({total_evaluated} receipts evaluated)")
    print("=" * 50)
    for field, correct in correct_counts.items():
        accuracy = (correct / total_evaluated * 100) if total_evaluated > 0 else 0
        print(f"{field:10s}: {correct}/{total_evaluated} correct  ({accuracy:.1f}%)")


if __name__ == "__main__":
    evaluate_batch(num_samples=20)