"""Load ArrivalCNN weights and run inference (CPU)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Same path as deployment; weights must match ArrivalCNN below.
MODEL_PATH = Path(__file__).resolve().parent.parent / "dataset" / "training.pth"
FALLBACK_MODEL_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "training.pth"
logger = logging.getLogger(__name__)

# Matches models/xgboost_model.ipynb (TARGET_LEN / fix_length before ArrivalCNN training).
SIGNAL_LEN = 6000


class ArrivalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.squeeze(-1))


def _to_signal_length(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size >= SIGNAL_LEN:
        return arr[:SIGNAL_LEN].copy()
    out = np.zeros(SIGNAL_LEN, dtype=np.float32)
    out[: arr.size] = arr
    return out


def _strip_module_prefix(state: dict) -> dict:
    if not state:
        return state
    keys = list(state.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[7:]: v for k, v in state.items()}
    return state


def load_model() -> ArrivalCNN:
    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    if not model_path.exists():
        message = (
            "Model file not found. Expected one of: "
            f"{MODEL_PATH.as_posix()} or {FALLBACK_MODEL_PATH.as_posix()}"
        )
        logger.error(message)
        raise FileNotFoundError(message)

    device = torch.device("cpu")
    checkpoint = torch.load(
        str(model_path),
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        model = ArrivalCNN()
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state = _strip_module_prefix(checkpoint["state_dict"])
            elif "model_state_dict" in checkpoint:
                state = _strip_module_prefix(checkpoint["model_state_dict"])
            else:
                state = _strip_module_prefix(checkpoint)
            model.load_state_dict(state)
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    model.to(device)
    model.eval()
    return model


def predict(model: ArrivalCNN, values: list[float]) -> list[float]:
    raw = _to_signal_length(values)
    x = torch.from_numpy(raw).view(1, 1, -1)
    with torch.no_grad():
        out = model(x)
    return out.squeeze(0).cpu().numpy().tolist()
