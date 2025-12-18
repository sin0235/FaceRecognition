import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch import optim

from models.facenet.facenet_model import FaceNetModel, TripletLoss
from models.facenet.facenet_dataloader import FaceNetTripletDataset


# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = r"D:\Github_HTDA\CelebA_Aligned_Balanced\train"

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
MARGIN = 0.3
PATIENCE = 2

SAVE_DIR = "models/checkpoints/facenet"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = os.path.join(SAVE_DIR, "training_log.txt")


# ================= LOG =================
def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ================= DATA =================
train_dataset = FaceNetTripletDataset(TRAIN_DIR)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)


# ================= MODEL =================
model = FaceNetModel(embedding_size=512).to(DEVICE)
criterion = TripletLoss(margin=MARGIN)
optimizer = optim.Adam(model.parameters(), lr=LR)


# ================= TRAIN =================
def train_one_epoch():
    model.train()
    total_loss = 0.0

    for i, (a, p, n) in enumerate(train_loader):
        a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)

        optimizer.zero_grad()
        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)

        loss = criterion(emb_a, emb_p, emb_n)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"  Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)


def main():
    best_loss = float("inf")
    patience_counter = 0

    log("=" * 60)
    log("START FACENET TRAINING")
    log(f"Start time: {datetime.now()}")
    log(f"Device: {DEVICE}")
    log("=" * 60)

    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch()

        log(
            f"[Epoch {epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save last
        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, "facenet_last.pth")
        )

        # Save best
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "facenet_best.pth")
            )
            log("Saved best model (facenet_best.pth)")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            log("Early stopping triggered")
            break

    total_time = time.time() - start_time
    log("=" * 60)
    log(f"TRAINING FINISHED in {total_time/3600:.2f} hours")
    log(f"Best training loss: {best_loss:.4f}")
    log("=" * 60)


if __name__ == "__main__":
    main()
