import xgboost as xgb
import pandas as pd
import joblib
import torch
import os

from sklearn.preprocessing import StandardScaler
from typing import Tuple

from trainer.config import Config as trainer_config
from trainer.train_nn import FraudNN


FRAUD_THRESHOLD: float = 0.02
ENSEMBLE_XGB_WEIGHT: float = 0.4

XGB_MODEL_PATH: str = "./model/xgboost_fraud.ubj"
MLP_MODEL_PATH: str = "./model/nn_fraud.pth"
SCALER_PATH: str = "./model/nn_scaler.pkl"

CSV_PATH: str = "../preprocessor/output/predict/predict_agg.csv"


def load_features(csv_path: str) -> Tuple[pd.DataFrame, pd.Index]:
    df = pd.read_csv(csv_path)
    accounts = df.iloc[:, 0]
    X = df[trainer_config.FEATURES]
    print(f"Loaded {len(df):,} rows")
    return X, accounts


def load_xgb(model_path: str) -> xgb.Booster:
    print(f"Loading XGB from {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def load_mlp(model_path: str, scaler_path: str) -> Tuple[FraudNN, StandardScaler]:
    print(f"Loading MLP model from {model_path} and scaler from {scaler_path}")
    scaler: StandardScaler = joblib.load(scaler_path)
    model = FraudNN(input_size=len(trainer_config.FEATURES))
    device = torch.device(trainer_config.NN_DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, scaler


def predict_xgb(csv_path: str, model_path: str) -> pd.DataFrame:
    print("[XGB] Loading Features")
    X, accounts = load_features(csv_path)

    print("[XGB] Loading XGB model")
    model = load_xgb(model_path)
    dmatrix = xgb.DMatrix(X)
    probs = model.predict(dmatrix)
    labels = (probs > FRAUD_THRESHOLD).astype(int)

    print(f"[XGB] Flagged: {labels.sum()} | Mean prob: {probs.mean():.6f}")
    return pd.DataFrame({
        "acct": accounts,
        "xgb_prob": probs,
        "label": labels
    })


def predict_mlp(csv_path: str, model_path: str, scaler_path: str) -> pd.DataFrame:
    print("[MLP] Loading features")
    X, accounts = load_features(csv_path)

    print("[MLP] Loading model and scaler")
    model, scaler = load_mlp(model_path, scaler_path)
    X_scaled = scaler.transform(X)

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(
            next(model.parameters()).device)
        probs = model(X_tensor).cpu().numpy().flatten()
    labels = (probs > FRAUD_THRESHOLD).astype(int)

    print(f"[MLP] Flagged: {labels.sum()} | Mean prob: {probs.mean():.6f}")
    return pd.DataFrame({
        "acct": accounts,
        "mlp_prob": probs,
        "label": labels
    })


def predict_ensemble(csv_path: str, xgb_path: str, mlp_path: str, scaler_path: str) -> pd.DataFrame:
    df_xgb = predict_xgb(csv_path, xgb_path)
    df_mlp = predict_mlp(csv_path, mlp_path, scaler_path)

    weight = ENSEMBLE_XGB_WEIGHT

    xgb_prob = df_xgb["xgb_prob"].values
    mlp_prob = df_mlp["mlp_prob"].values

    ensemble_probs = weight * xgb_prob + (1 - weight) * mlp_prob
    labels = (ensemble_probs > FRAUD_THRESHOLD).astype(int)

    print(f"[ENSEMBLE] {weight:.1f}*XGB + {(1-weight):.1f}*NN")
    print(f"[ENSEMBLE] Flagged: {labels.sum()} | Mean prob: {
          ensemble_probs.mean():.6f}")

    return pd.DataFrame({
        "acct": df_xgb["acct"],
        "label": labels
    })


def save(df: pd.DataFrame, name: str):
    os.makedirs("output", exist_ok=True)
    path = f"output/{name}.csv"
    submission = df[["acct", "label"]].copy()
    submission.to_csv(path, index=False)
    print(f"Saved {path} | Flagged: {submission["label"].sum()}")


def main() -> None:
    df_xgb = predict_xgb(CSV_PATH, XGB_MODEL_PATH)
    save(df_xgb, "xgb_predictions")

    print()

    df_nn = predict_mlp(CSV_PATH, MLP_MODEL_PATH, SCALER_PATH)
    save(df_nn, "nn_predictions")

    print()

    df_ens = predict_ensemble(CSV_PATH, XGB_MODEL_PATH,
                              MLP_MODEL_PATH, SCALER_PATH)
    save(df_ens, "ensemble_predictions")


if __name__ == "__main__":
    main()
