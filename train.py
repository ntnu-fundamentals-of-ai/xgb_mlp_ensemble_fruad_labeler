import os

from trainer.evaluator import evaluate_model
from trainer.data_loader import load_data
from trainer.train_xgb import train_xgb
from trainer.train_nn import train_nn
from trainer.config import Config


def main() -> None:
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    print("Loading Data...")
    training, validate, testing = load_data()

    print(f"Train: {len(training)} | Fraud: {training['label'].sum()}")
    print(f"Val:   {len(validate)} | Fraud: {validate['label'].sum()}")
    print(f"Test:  {len(testing)} | Fraud: {testing['label'].sum()}")

    print("\nTraining XGBoost...")

    xgb_model = train_xgb(training, validate)

    print("\nEvaluating...")
    results = evaluate_model(xgb_model, testing)
    print(f"Test AUC: {results["auc"]:.4f}")

    print("\nSaving model...")
    xgb_model.save_model(f"{Config.MODEL_DIR}/xgboost_fraud.ubj")

    print("Training Neural Network")
    _, _ = train_nn()


if __name__ == "__main__":
    main()
