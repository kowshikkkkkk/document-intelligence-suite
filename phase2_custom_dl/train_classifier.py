"""
Phase 2, Step 3: Train a custom PyTorch classifier on the embedded chunks.

This is the core deep learning piece: a hand-written nn.Module architecture
and a hand-written training loop (forward pass, loss, backward, optimizer
step) -- no high-level Trainer abstraction, so every step is visible and
understood.

Input: frozen sentence-transformer embeddings (384-dim vectors)
Output: classification into company / date / total / other
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

EMBEDDINGS_PATH = Path(__file__).parent / "embeddings.npz"
MODEL_SAVE_PATH = Path(__file__).parent / "classifier.pt"

LABEL_NAMES = ["company", "date", "total", "other"]


# --- Step A: PyTorch Dataset wrapper ---
class ChunkDataset(Dataset):
    """Wraps our numpy embeddings + labels so PyTorch's DataLoader can
    batch and shuffle them automatically during training."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# --- Step B: The model itself ---
class ChunkClassifier(nn.Module):
    """Simple feedforward network (MLP):
    384-dim input vector -> hidden layer -> 4-class output.
    """

    def __init__(self, input_dim=384, hidden_dim=128, num_classes=4):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weighting, so rare classes (company/date/total)
    are penalized more heavily than the majority class (other) during training."""
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / (counts * num_classes)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, data_loader, criterion):
    """Runs the model on a dataset WITHOUT updating weights -- used for
    checking validation performance during training."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # torch.no_grad() disables gradient tracking since we're not training here
    with torch.no_grad():
        for vectors, labels in data_loader:
            outputs = model(vectors)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * vectors.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train():
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    data = np.load(EMBEDDINGS_PATH)
    embeddings, labels = data["embeddings"], data["labels"]

    full_dataset = ChunkDataset(embeddings, labels)

    # 80/20 train/validation split, reproducible via fixed seed
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train size: {train_size}, Validation size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ChunkClassifier(input_dim=384, hidden_dim=128, num_classes=4)

    class_weights = compute_class_weights(labels, num_classes=4)
    print(f"Class weights: {dict(zip(LABEL_NAMES, class_weights.tolist()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30
    patience = 5  # stop if val loss doesn't improve for this many epochs in a row
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    print(f"\nTraining for up to {num_epochs} epochs (early stopping patience={patience})...\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for vectors, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(vectors)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * vectors.size(0)

        train_loss /= train_size
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()  # save a snapshot of the BEST weights so far
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nNo improvement for {patience} epochs -- stopping early at epoch {epoch+1}.")
                break

    # Restore the best model seen during training, not just whatever the last epoch produced
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()