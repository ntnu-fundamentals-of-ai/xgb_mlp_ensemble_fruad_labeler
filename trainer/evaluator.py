import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from typing import Dict
from .config import Config


def evaluate_model(model: xgb.XGBClassifier, testing: pd.DataFrame) -> Dict[str, object]:
    X_testing, y_testing = testing[Config.FEATURES], testing["label"]
    y_predict_probability = model.predict_proba(X_testing)[:, 1]
    y_predict = (y_predict_probability >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y_testing, y_predict_probability),
        "report:": classification_report(y_testing, y_predict, output_dict=True),
        "cm": confusion_matrix(y_testing, y_predict),
        "y_predict_Probability": y_predict_probability,
        "y_testing": y_testing
    }
