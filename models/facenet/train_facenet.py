import torch
from torch.utils.data import DataLoader
from torch import optim
from models.facenet.facenet_model import FaceNetModel, TripletLoss
from models.facenet.facenet_dataloader import FaceNetTripletDataset
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/CelebA_Aligned_Balanced/train"
SAVE_PATH = "models/checkpoints/facenet/facenet_best.pth"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# Dataset & Loader
dataset = FaceNetTripletDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

# Model
model = FaceNetModel(embedding_size=512).to(DEVICE)
criterion = TripletLoss(margin=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train
model.train()
for epoch in range(2):
    total_loss = 0
    for i, (a, p, n) in enumerate(loader):
        a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
        # Zero gradients
        optimizer.zero_grad()

        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)
        # Compute loss
        loss = criterion(emb_a, emb_p, emb_n)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"[Epoch {epoch}] Step {i} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch} Avg Loss: {total_loss / len(loader):.4f}")

# Save model
torch.save(model.state_dict(), SAVE_PATH)
print("Saved FaceNet model to:", SAVE_PATH)
