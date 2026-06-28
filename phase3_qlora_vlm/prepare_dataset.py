"""
Phase 3, Step 1: Prepare the SROIE dataset in conversational JSONL format
for QLoRA fine-tuning of Qwen2.5-VL-3B-Instruct.

Each line in the output JSONL is one training example structured as a
chat exchange: user provides an image + instruction, assistant responds
with the target JSON. This mirrors how the model will be used at
inference time after fine-tuning.

This script only prepares the data -- actual training happens on Kaggle,
since it needs GPU memory this machine doesn't have.
"""

import json
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent / "data" / "raw" / "SROIE2019"
TRAIN_ROOT = DATA_ROOT / "train"
IMG_DIR = TRAIN_ROOT / "img"
ENTITIES_DIR = TRAIN_ROOT / "entities"

OUTPUT_PATH = Path(__file__).parent / "vlm_train_data.jsonl"

# Fixed instruction used consistently across every training example.
# Keeping this exact wording at inference time too is what makes the
# fine-tuned behavior reliable.
INSTRUCTION = (
    "Extract the company name, date, and total amount from this receipt. "
    "Respond only with valid JSON in this exact format: "
    '{"company": "...", "date": "...", "total": "..."}'
)


def load_ground_truth(receipt_id: str) -> dict:
    label_path = ENTITIES_DIR / f"{receipt_id}.txt"
    raw_text = label_path.read_text(encoding="utf-8", errors="ignore")
    return json.loads(raw_text)


def build_example(receipt_id: str, image_path: Path) -> dict:
    truth = load_ground_truth(receipt_id)

    # Only keep the 3 fields we care about, even though SROIE's ground
    # truth also includes "address" -- we deliberately scoped that out
    # back in Phase 0.
    target_json = json.dumps({
        "company": truth.get("company", ""),
        "date": truth.get("date", ""),
        "total": truth.get("total", ""),
    })

    return {
        "receipt_id": receipt_id,
        "image_path": str(image_path),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": INSTRUCTION},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_json},
                ],
            },
        ],
    }


def build_dataset(limit: int = 150):
    all_ids = sorted(p.stem for p in IMG_DIR.glob("*.jpg"))
    if limit:
        all_ids = all_ids[:limit]

    examples = []
    skipped = 0

    for receipt_id in all_ids:
        # Store a relative path (relative to SROIE2019/), not an absolute
        # Windows path -- this keeps the JSONL portable so it still makes
        # sense once we're running on Kaggle's filesystem, not just locally.
        img_path = (Path("train") / "img" / f"{receipt_id}.jpg").as_posix()      
        try:
            example = build_example(receipt_id, img_path)
            examples.append(example)
        except Exception as e:
            print(f"Skipped {receipt_id}: {e}")
            skipped += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Built {len(examples)} examples (skipped {skipped})")
    print(f"Saved to {OUTPUT_PATH}")

    # Show one example so we can visually sanity-check the structure
    print("\n--- Sample example ---")
    print(json.dumps(examples[0], indent=2))


if __name__ == "__main__":
    build_dataset(limit=150)