import xgboost as xgb
import pandas as pd

from .config import Config


def train_xgb(training: pd.DataFrame, validate: pd.DataFrame) -> xgb.XGBClassifier:
    X_training, y_training = training[Config.FEATURES], training["label"]
    X_Validate, y_validate = validate[Config.FEATURES], validate["label"]

    params = {k: v for k, v in Config.XGB_PARAMS.items() if k !=
              "scale_pos_weight"}

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_training, y_training,
        eval_set=[(X_Validate, y_validate)],
        verbose=10
    )

    return model
