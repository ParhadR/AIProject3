import os
import torch
import pandas as pd
import numpy as np
from train_model import MLP

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "model_T_ship1.pt"  # Change to test other models
MODEL_PATH = os.path.join("models", MODEL_NAME)
DATA_PATH = "data/T_ship1.csv"
NUM_SAMPLES = 10

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
# Report
# -----------------------------
print(f"\nPredictions from model: {MODEL_NAME}")
for i in range(NUM_SAMPLES):
    coords = samples.iloc[i][['bx', 'by', 'rx', 'ry']].tolist()
    print(f"Input: {coords}  |  True T: {true_T[i]:.2f}  |  Predicted T: {preds[i]:.2f}")
