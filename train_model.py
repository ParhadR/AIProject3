import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------
# Custom Dataset
# -------------------------
class TDataset(Dataset):
    def __init__(self, df):
        inputs = df[["bx", "by", "rx", "ry"]].values
        targets = df["T"].values

        self.X = torch.tensor(inputs, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# Model Definition
# -------------------------
class MLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------
# Training Function
# -------------------------
def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for X, y in loader:
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)

# -------------------------
# Evaluation Function
# -------------------------
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)

# -------------------------
# Main Script
# -------------------------
def main(csv_path="data/T_ship1.csv", epochs=50, batch_size=128, lr=1e-3):
    df = pd.read_csv(csv_path)

    # Normalize inputs (optional)
    df[["bx", "by", "rx", "ry"]] /= df[["bx", "by", "rx", "ry"]].max()

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_data = TDataset(train_df)
    val_data = TDataset(val_df)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate(model, val_loader, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save model to models/ folder
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "model_T_ship1.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Plot training/validation loss
    os.makedirs("data", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/training_curve.png")
    plt.show()

if __name__ == "__main__":
    main()
