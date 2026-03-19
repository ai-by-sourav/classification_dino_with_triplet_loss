import json
from types import SimpleNamespace

REQUIRED_KEYS = [
    "train_dir", "val_dir", "class_names", "epochs", "learning_rate",
    "batch_size", "patience", "optimizer", "result_dir",
    "use_hard_triplet_loss", "use_soft_triplet_loss", "triplet_weight",
    "model_name", "model_repo"
]

DEFAULTS = {
    "epochs": 20,
    "learning_rate": 1e-4,
    "batch_size": 32,
    "patience": 5,
    "optimizer": "adam",
    "result_dir": "./outputs",
    "use_hard_triplet_loss": False,
    "use_soft_triplet_loss": False,
    "triplet_weight": 0.5
}


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        raw = json.load(f)

    for key in ["train_dir", "val_dir", "class_names", "model_name", "model_repo"]:
        if key not in raw:
            raise ValueError(f"Missing required config key: '{key}'")

    
    for key, val in DEFAULTS.items():
        raw.setdefault(key, val)

    if raw["optimizer"] not in ("adam", "adamw"):
        raise ValueError(f"optimizer must be 'adam' or 'adamw', got '{raw['optimizer']}'")

    
    if raw["use_hard_triplet_loss"] and raw["use_soft_triplet_loss"]:
        raise ValueError("Only one triplet loss type can be active at a time.")

    if not isinstance(raw["class_names"], list) or len(raw["class_names"]) == 0:
        raise ValueError("class_names must be a non-empty list.")

    return SimpleNamespace(**raw)