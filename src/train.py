import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import mlflow
from dataset import CatsDogsDataset
from model import build_model
import os, json

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model():
    mlflow.start_run()

    aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    dataset = CatsDogsDataset("data/train", transform=aug)

    train_len = int(len(dataset) * 0.8)
    val_len = int(len(dataset) * 0.1)
    test_len = len(dataset) - train_len - val_len

    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        mlflow.log_metric("loss", total_loss/len(train_loader), step=epoch)
        print(f"Epoch {epoch+1}: {total_loss/len(train_loader):.4f}")

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pt")
    mlflow.log_artifact("artifacts/model.pt")
    mlflow.end_run()

if __name__ == "__main__":
    train_model()