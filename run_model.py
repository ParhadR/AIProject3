import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train_model import MLP

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "model_T_ship1.pt"  # Change to test other models
MODEL_PATH = os.path.join("models", MODEL_NAME)
DATA_PATH = "data/T_ship1.csv"
NUM_SAMPLES = 1000  # More samples = better graph

# -----------------------------
# Load and Normalize Data
# -----------------------------
df = pd.read_csv(DATA_PATH)
df_norm = df.copy()
df_norm[["bx", "by", "rx", "ry"]] /= df_norm[["bx", "by", "rx", "ry"]].max()

samples = df.sample(NUM_SAMPLES, random_state=42).reset_index(drop=True)
inputs = df_norm.loc[samples.index, ["bx", "by", "rx", "ry"]].values
true_T = df.loc[samples.index, "T"].values

# -----------------------------
# Load Model
# -----------------------------
model = MLP()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# -----------------------------
# Predict
# -----------------------------
with torch.no_grad():
    preds = model(torch.tensor(inputs, dtype=torch.float32)).squeeze().numpy()

# -----------------------------
# Report Sample Predictions
# -----------------------------
print(f"\nPredictions from model: {MODEL_NAME}")
for i in range(min(10, NUM_SAMPLES)):
    coords = samples.iloc[i][['bx', 'by', 'rx', 'ry']].tolist()
    print(f"Input: {coords}  |  True T: {true_T[i]:.2f}  |  Predicted T: {preds[i]:.2f}")

# -----------------------------
# Plot True vs Predicted
# -----------------------------
plt.figure(figsize=(6, 6))
plt.scatter(true_T, preds, alpha=0.5, s=15)
plt.plot([true_T.min(), true_T.max()], [true_T.min(), true_T.max()], 'r--')  # Ideal line
plt.xlabel("True T")
plt.ylabel("Predicted T")
plt.title("Predicted vs True T Values")
plt.grid(True)
plt.tight_layout()
os.makedirs("data", exist_ok=True)
plt.savefig("data/pred_vs_true_T.png")
plt.show()
