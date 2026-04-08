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
# Config defaults
# ---------------------------------------------------------------------------
TASKS = ["activity_recognition", "knife_sharpness", "boning_vs_slicing"]
PAD_TARGET_LEN = int(os.getenv("BENTOML_PAD_LEN", "60"))

# ---------------------------------------------------------------------------
# Model & artifact loading helpers
# ---------------------------------------------------------------------------
def _load_task_bundle(task_name: str):
    """
    Scan the outputs/ directory for the most recent best_model.pt for a specific task.
    Prefers 'bilstm' models. Allows graceful fallback to whatever architecture is available.
    """
    from model.bilstm import BiLSTM
    from model.gru import GRU
    from model.tcn import TCN

    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError("No outputs/ directory found. Train a model first.")

    candidates = sorted(outputs_dir.rglob("best_model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No best_model.pt found in outputs/.")

    selected_ckpt = None
    
    # Pass 1: Look for exact task + bilstm
    for p in candidates:
        try:
            checkpoint = torch.load(p, map_location="cpu", weights_only=False)
            cfg = checkpoint.get("config", {})            
            t_name = cfg.get("task", {}).get("name", "")
            m_type = cfg.get("model", {}).get("type", "").lower()
            if t_name == task_name and m_type == "bilstm":
                selected_ckpt = checkpoint
                print(f"[BentoML] Loading {m_type} for {task_name} from: {p}")
                break
        except Exception:
            continue
            
    # Pass 2: If no bilstm, take the most recent model for this task
    if not selected_ckpt:
        for p in candidates:
            try:
                checkpoint = torch.load(p, map_location="cpu", weights_only=False)
                cfg = checkpoint.get("config", {})
                t_name = cfg.get("task", {}).get("name", "")
                if t_name == task_name:
                    m_type = cfg.get("model", {}).get("type", "").lower()
                    selected_ckpt = checkpoint
                    print(f"[BentoML] Fallback Loading {m_type} for {task_name} from: {p}")
                    break
            except Exception:
                continue
                
    if not selected_ckpt:
        raise FileNotFoundError(f"No checkpoint found for task: {task_name}")

    cfg = selected_ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    mtype = model_cfg.get("type", "bilstm").lower()
    hidden_size = model_cfg.get("hidden_size", 128)
    num_layers = model_cfg.get("num_layers", 1)
    dropout = model_cfg.get("dropout", 0.4)

    label_encoder = selected_ckpt.get("label_encoder")
    scaler = selected_ckpt.get("scaler")
    
    if label_encoder is None or scaler is None:
        raise ValueError(f"Checkpoint for {task_name} is missing scaler or label_encoder.")

    num_classes = len(label_encoder.classes_)
    input_size = scaler.n_features_in_

    # Build model
    if mtype == "gru":
        model = GRU(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, num_layers=num_layers, dropout=dropout)
    elif mtype == "tcn":
        model = TCN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, num_layers=num_layers, dropout=dropout)
    else:
        model = BiLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, num_layers=num_layers, dropout=dropout)

    state_dict = selected_ckpt.get("model_state_dict", selected_ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "num_classes": num_classes,
        "input_features": input_size,
        "model_type": mtype
    }

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
    BentoML service that hosts multiple PyTorch models for different tasks.
    """
    def __init__(self):
        print("[BentoML] Initializing Multi-task MotionClassifier service...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_bundles = {}

        for task in TASKS:
            try:
                bundle = _load_task_bundle(task)
                bundle["model"].to(self.device)
                self.task_bundles[task] = bundle
                print(f"[BentoML] ✅ {task} ready -> {bundle['model_type']} ({bundle['num_classes']} classes, {bundle['input_features']} features)")
            except Exception as e:
                print(f"[BentoML] ⚠️ Failed to load task {task}: {e}")

    @bentoml.api()
    def predict(self, input_array: np.ndarray, task: str = "activity_recognition") -> dict[str, Any]:
        """
        Predict the class for a single motion window.
        """
        if task not in self.task_bundles:
            return {"error": f"Task '{task}' is not loaded. Available: {list(self.task_bundles.keys())}"}
            
        bundle = self.task_bundles[task]
        
        if input_array.ndim != 2:
            return {"error": f"Expected 2D array, got shape {input_array.shape}"}

        if input_array.shape[1] != bundle["input_features"]:
            return {"error": f"Feature mismatch: model expects {bundle['input_features']} features, got {input_array.shape[1]}"}

        scaled = bundle["scaler"].transform(input_array.astype(np.float32))
        padded = _pad_or_truncate(scaled, PAD_TARGET_LEN)
        tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = bundle["model"](tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = bundle["label_encoder"].inverse_transform([pred_idx])[0]

        probabilities = {
            str(cls): round(float(p), 4)
            for cls, p in zip(bundle["label_encoder"].classes_, probs)
        }

        return {
            "prediction": str(pred_label),
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": probabilities,
        }

    @bentoml.api()
    def predict_batch(self, input_arrays: np.ndarray, task: str = "activity_recognition") -> dict[str, Any]:
        if input_arrays.ndim != 3:
            return {"error": f"Expected 3D array (batch, seq_len, features), got shape {input_arrays.shape}"}

        results = []
        for i in range(input_arrays.shape[0]):
            results.append(self.predict(input_arrays[i], task=task))

        return {"predictions": results, "batch_size": len(results)}

    @bentoml.api()
    def model_info(self) -> dict[str, Any]:
        """Return metadata about all loaded models."""
        info = {}
        for tsk, bundle in self.task_bundles.items():
            info[tsk] = {
                "model_type": bundle["model_type"],
                "num_classes": bundle["num_classes"],
                "class_labels": [str(c) for c in bundle["label_encoder"].classes_],
                "input_features": bundle["input_features"],
                "pad_target_len": PAD_TARGET_LEN
            }
        
        return {
            "device": str(self.device),
            "tasks": info,
        }
