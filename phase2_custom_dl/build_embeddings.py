"""
Phase 2, Step 2: Convert labeled OCR text chunks into embedding vectors.

Loads labeled_chunks.json (text + label pairs), runs each text chunk
through a pretrained sentence-transformer to get a fixed-size vector,
and saves everything together as a .npz file -- ready for PyTorch's
Dataset/DataLoader to consume in the next step.

We use a FROZEN pretrained embedding model here (not training it) --
only our own classifier head (built in the next step) gets trained.
This mirrors a common real-world pattern: leverage a pretrained
embedding model, train a small task-specific head on top of it.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path(__file__).parent / "labeled_chunks.json"
OUTPUT_PATH = Path(__file__).parent / "embeddings.npz"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Fixed order so label <-> integer mapping is consistent everywhere downstream
LABEL_TO_ID = {"company": 0, "date": 1, "total": 2, "other": 3}


def build_embeddings():
    print(f"Loading labeled data from {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        labeled_rows = json.load(f)

    print(f"Loaded {len(labeled_rows)} labeled chunks")

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [row["text"] for row in labeled_rows]
    labels = [LABEL_TO_ID[row["label"]] for row in labeled_rows]

    print("Encoding texts into vectors (this may take a minute)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"\nEmbeddings shape: {embeddings.shape}")  # (num_samples, 384)
    print(f"Labels shape: {labels.shape}")

    np.savez(OUTPUT_PATH, embeddings=embeddings, labels=labels)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_embeddings()