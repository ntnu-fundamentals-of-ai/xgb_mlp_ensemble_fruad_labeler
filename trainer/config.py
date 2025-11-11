import torch

from dataclasses import dataclass
from typing import List, Any, Dict


@dataclass
class Config:
    DATA_DIR: str = "../preprocessor/output/agg"
    MODEL_DIR: str = "model"

    SEED: int = 42
    POS_WEIGHT_RATIO: float = 1.0

    FEATURES: List[str] | None = None
    XGB_PARAMS: Dict[str, Any] | None = None

    NN_HIDDEN: List[int] | None = None
    NN_DROPOUT: float = 0.2
    NN_LR: float = 0.0002
    NN_WEIGHT_DECAY: float = 1e-4
    NN_EPOCHS: int = 1_000
    NN_PATIENCE: int = 30
    NN_BATCH_SIZE: int = 1024
    NN_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


Config.NN_HIDDEN = [256, 128, 64]


Config.XGB_PARAMS = {
    "n_estimators": 3000,
    "max_depth": 9,
    "learning_rate": 0.02,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "min_child_weight": 1,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "auc",
    "tree_method": "hist",
    "early_stopping_rounds": 100,
}

Config.FEATURES = [
    "total_txns", "incoming_txns", "outgoing_txns", "self_txns",
    "cross_bank_txns", "total_in_amt", "total_out_amt", "avg_txn_amt",
    "max_txn_amt", "std_txn_amt", "unique_dates", "txns_per_day",
    "time_span_days", "night_txns", "channel_1", "channel_2",
    "channel_3", "channel_4", "channel_5", "channel_6",
    "channel_7", "channel_99", "channel_unk", "unique_currencies",
    "curr_top1", "curr_top2", "curr_top3", "curr_top4",
    "curr_other", "high_amt_ratio", "scheduled_ratio", "sin_time_5min",
    "cos_time_5min", "small_out_count", "small_out_ratio", "small_out_avg_amt",
    "velocity_ratio", "high_amt_burst_ratio",
    "burst_day_count", "burst_txn_total", "burst_amt_total",
    "burst_txn_ratio", "burst_amt_ratio"
]
