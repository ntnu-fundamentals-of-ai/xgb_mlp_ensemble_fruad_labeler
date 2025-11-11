import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .data_loader import load_data
from typing import Tuple
from .config import Config


class FraudNN(nn.Module):
    def __init__(
        self,
        input_size: int,
    ) -> None:
        super().__init__()
        hidden1, hidden2, hidden3 = Config.NN_HIDDEN
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(Config.NN_DROPOUT),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(Config.NN_DROPOUT),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(Config.NN_DROPOUT),

            nn.Linear(hidden3, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_nn() -> Tuple[FraudNN, StandardScaler]:
    print("Loading data for NN training...")
    training, validate, _ = load_data()
    df = pd.concat([training, validate], ignore_index=True)

    X = df[Config.FEATURES].copy()
    y = df["label"].copy()

    print(f"Training NN on {len(df):,} samples | Fraud: {y.sum():,}")

    X_training, X_validate, y_training, y_validate = train_test_split(
        X, y, test_size=0.2, random_state=Config.SEED, stratify=y
    )

    scaler = StandardScaler()
    X_training_scaled = scaler.fit_transform(X_training)
    X_validate_scaled = scaler.transform(X_validate)

    X_training_t = torch.FloatTensor(X_training_scaled).to(Config.NN_DEVICE)
    X_validate_t = torch.FloatTensor(X_validate_scaled).to(Config.NN_DEVICE)

    y_training_t = torch.FloatTensor(
        y_training.values).reshape(-1, 1).to(Config.NN_DEVICE)
    y_validate_t = torch.FloatTensor(
        y_validate.values).reshape(-1, 1).to(Config.NN_DEVICE)

    model = FraudNN(
        input_size=len(Config.FEATURES),
    ).to(Config.NN_DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.NN_LR,
        weight_decay=Config.NN_WEIGHT_DECAY
    )

    print("Starting training...")
    model.train()
    best_validate_loss = float("inf")
    patience_counter = 0
    best_model_path = f"{Config.MODEL_DIR}/nn_fraud_best.pth"

    for epoch in range(1, Config.NN_EPOCHS + 1):
        optimizer.zero_grad()
        outputs = model(X_training_t)
        loss = criterion(outputs, y_training_t)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                validate_outputs = model(X_validate_t)
                validate_loss = criterion(
                    validate_outputs, y_validate_t).item()
            model.train()

            print(f"Epoch {epoch:3d} | Training Loss: {
                  loss.item():.6f} | Validate loss: {validate_loss:.6f}")

            if validate_loss < best_validate_loss:
                best_validate_loss = validate_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= Config.NN_PATIENCE:
                    print(f"Early stop triggered at epoch {epoch}")
                    break
    print(f"Loading best model with validate loss: {best_validate_loss:.6f}")
    model.load_state_dict(torch.load(
        best_model_path, map_location=Config.NN_DEVICE))

    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    final_model_path = f"{Config.MODEL_DIR}/nn_fraud.pth"
    scaler_path = f"{Config.MODEL_DIR}/nn_scaler.pkl"

    torch.save(model.state_dict(), final_model_path)
    joblib.dump(scaler, scaler_path)

    print(f"NN Saved: {final_model_path}")
    print(f"Scaler Saved: {scaler_path}")

    return model, scaler
