"""
BentoML Service for Motion Classification.

Loads the best PyTorch model from MLflow Model Registry and serves
predictions over HTTP.

Start with:
    bentoml serve serving.service:MotionClassifier --reload

Endpoints:
    POST /predict  — single window inference
    POST /predict_batch — multi-window batch inference
    GET  /model_info — current model metadata
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import bentoml
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config — defaults can be overridden via environment variables
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{Path(__file__).resolve().parent.parent / 'mlflow_tracking.db'}",
)
TASK_NAME = os.getenv("BENTOML_TASK_NAME", "activity_recognition")
MODEL_STAGE = os.getenv("BENTOML_MODEL_STAGE", "latest")  # "latest" or "Production"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "shared"
PAD_TARGET_LEN = int(os.getenv("BENTOML_PAD_LEN", "60"))


# ---------------------------------------------------------------------------
# Model & artifact loading helpers
# ---------------------------------------------------------------------------
def _load_mlflow_model():
    """Load the PyTorch model from the MLflow registry."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{TASK_NAME}/{MODEL_STAGE}"
    print(f"[BentoML] Loading model from MLflow: {model_uri}")

    try:
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        print(f"[BentoML] ✅ Model loaded successfully ({type(model).__name__})")
        return model
    except Exception as exc:
        print(f"[BentoML] ⚠️  MLflow registry load failed ({exc}). Falling back to latest checkpoint...")
        return _load_fallback_model()


def _load_fallback_model():
    """
    Fallback: scan the outputs/ directory for the most recent best_model.pt
    and load it using the bundled config to reconstruct the architecture.
    """
    from model.bilstm import BiLSTM
    from model.gru import GRU
    from model.tcn import TCN

    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(
            "No outputs/ directory found. Train a model first with `python train.py`."
        )

    # Find the most recent best_model.pt
    candidates = sorted(outputs_dir.rglob("best_model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No best_model.pt found in outputs/. Train a model first.")

    ckpt_path = candidates[0]
    print(f"[BentoML] Loading fallback checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    mtype = model_cfg.get("type", "bilstm").lower()
    hidden_size = model_cfg.get("hidden_size", 128)
    num_layers = model_cfg.get("num_layers", 1)
    dropout = model_cfg.get("dropout", 0.4)

    # Determine input_size and num_classes from saved artifacts
    label_encoder = checkpoint.get("label_encoder")
    scaler = checkpoint.get("scaler")

    if label_encoder is not None:
        num_classes = len(label_encoder.classes_)
    else:
        # Try from shared artifacts
        le_path = ARTIFACTS_DIR / "label_encoder.pkl"
        if le_path.exists():
            with open(le_path, "rb") as f:
                label_encoder = pickle.load(f)
            num_classes = len(label_encoder.classes_)
        else:
            raise FileNotFoundError("Cannot determine num_classes. No label_encoder found.")

    if scaler is not None:
        input_size = scaler.n_features_in_
    else:
        sc_path = ARTIFACTS_DIR / "scaler.pkl"
        if sc_path.exists():
            with open(sc_path, "rb") as f:
                scaler = pickle.load(f)
            input_size = scaler.n_features_in_
        else:
            raise FileNotFoundError("Cannot determine input_size. No scaler found.")

    # Build model
    if mtype == "gru":
        model = GRU(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, num_layers=num_layers, dropout=dropout)
    elif mtype == "tcn":
        model = TCN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, num_layers=num_layers, dropout=dropout)
    else:
        model = BiLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, num_layers=num_layers, dropout=dropout)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[BentoML] ✅ Fallback model loaded ({type(model).__name__}, classes={num_classes}, features={input_size})")
    return model


def _load_scaler():
    """Load the StandardScaler from shared artifacts."""
    sc_path = ARTIFACTS_DIR / "scaler.pkl"
    if sc_path.exists():
        with open(sc_path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"scaler.pkl not found at {sc_path}")


def _load_label_encoder():
    """Load the LabelEncoder from shared artifacts."""
    le_path = ARTIFACTS_DIR / "label_encoder.pkl"
    if le_path.exists():
        with open(le_path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"label_encoder.pkl not found at {le_path}")


def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad (zero) or truncate a 2D array along axis 0 to target_len."""
    T, F = arr.shape
    if T >= target_len:
        return arr[:target_len]
    padded = np.zeros((target_len, F), dtype=arr.dtype)
    padded[:T] = arr
    return padded


# ---------------------------------------------------------------------------
# BentoML Service
# ---------------------------------------------------------------------------
@bentoml.service(
    name="motion_classifier",
    traffic={"timeout": 30},
    resources={"cpu": "1"},
)
class MotionClassifier:
    """
    BentoML service that wraps your PyTorch motion-classification model.

    Input:  a 2D NumPy array of shape (sequence_length, num_features)
    Output: predicted class label + confidence scores
    """

    def __init__(self):
        print("[BentoML] Initializing MotionClassifier service...")
        self.model = _load_mlflow_model()
        self.scaler = _load_scaler()
        self.label_encoder = _load_label_encoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.num_classes = len(self.label_encoder.classes_)
        self.input_features = self.scaler.n_features_in_

        print(
            f"[BentoML] Service ready — "
            f"model={type(self.model).__name__}, "
            f"classes={list(self.label_encoder.classes_)}, "
            f"features={self.input_features}, "
            f"device={self.device}"
        )

    # ---- Single window prediction -----------------------------------------
    @bentoml.api()
    def predict(self, input_array: np.ndarray) -> dict[str, Any]:
        """
        Predict the class for a single motion window.

        Parameters
        ----------
        input_array : np.ndarray
            Shape (sequence_length, num_features). Raw feature values
            (pre-scaling is handled internally).

        Returns
        -------
        dict with keys:
            prediction  : str   — predicted class label
            confidence  : float — softmax probability of the top class
            probabilities : dict — {class_name: probability} for every class
        """
        # Validate input dimensions
        if input_array.ndim != 2:
            return {"error": f"Expected 2D array (seq_len, features), got shape {input_array.shape}"}

        if input_array.shape[1] != self.input_features:
            return {
                "error": (
                    f"Feature mismatch: model expects {self.input_features} features, "
                    f"got {input_array.shape[1]}"
                )
            }

        # Preprocess: scale → pad → tensor
        scaled = self.scaler.transform(input_array.astype(np.float32))
        padded = _pad_or_truncate(scaled, PAD_TARGET_LEN)
        tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]

        probabilities = {
            str(cls): round(float(p), 4)
            for cls, p in zip(self.label_encoder.classes_, probs)
        }

        return {
            "prediction": str(pred_label),
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": probabilities,
        }

    # ---- Batch prediction -------------------------------------------------
    @bentoml.api()
    def predict_batch(self, input_arrays: np.ndarray) -> dict[str, Any]:
        """
        Predict classes for a batch of windows.

        Parameters
        ----------
        input_arrays : np.ndarray
            Shape (batch_size, sequence_length, num_features).

        Returns
        -------
        dict with key "predictions" containing a list of per-window results.
        """
        if input_arrays.ndim != 3:
            return {"error": f"Expected 3D array (batch, seq_len, features), got shape {input_arrays.shape}"}

        results = []
        for i in range(input_arrays.shape[0]):
            results.append(self.predict(input_arrays[i]))

        return {"predictions": results, "batch_size": len(results)}

    # ---- Model metadata ---------------------------------------------------
    @bentoml.api()
    def model_info(self) -> dict[str, Any]:
        """Return metadata about the currently loaded model."""
        return {
            "model_type": type(self.model).__name__,
            "task": TASK_NAME,
            "model_stage": MODEL_STAGE,
            "num_classes": self.num_classes,
            "class_labels": [str(c) for c in self.label_encoder.classes_],
            "input_features": self.input_features,
            "pad_target_len": PAD_TARGET_LEN,
            "device": str(self.device),
        }
