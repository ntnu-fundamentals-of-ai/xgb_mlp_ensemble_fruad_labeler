import pandas as pd

from typing import Tuple
from .config import Config


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training = pd.read_csv(f"{Config.DATA_DIR}/training_agg.csv")
    validate = pd.read_csv(f"{Config.DATA_DIR}/validate_agg.csv")
    testing = pd.read_csv(f"{Config.DATA_DIR}/testing_agg.csv")
    return training, validate, testing
