"""
End-to-end video inference orchestrator.

Chains all stages:
    Video → MediaPipe 2D → VideoPose3D 3D → Joint Mapping →
    Feature Engineering → Windowing → BiLSTM Prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data.preprocessing import (
    merge_sensors,
    engineer_features,
    create_windows,
    pad_windows_to_60,
    clean_features,
)
from model.bilstm import BiLSTM
from model.gru import GRU
from model.tcn import TCN

from .video_to_2d import extract_2d_poses
from .pose_2d_to_3d import lift_to_3d
from .joint_mapper import map_h36m_to_xsens


# ─────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────

def _load_bilstm_checkpoint(checkpoint_path: str | Path) -> dict:
    """
    Load a training checkpoint (``best_model.pt``).

    Returns a dict with keys: model, scaler, label_encoder, config.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    mtype = model_cfg.get("type", "bilstm").lower()
    hidden_size = model_cfg.get("hidden_size", 128)
    num_layers = model_cfg.get("num_layers", 1)
    dropout = model_cfg.get("dropout", 0.4)

    label_encoder: LabelEncoder = ckpt["label_encoder"]
    scaler: StandardScaler = ckpt["scaler"]

    num_classes = len(label_encoder.classes_)
    input_size = scaler.n_features_in_

    # Build model
    if mtype == "gru":
        model = GRU(input_size=input_size, hidden_size=hidden_size,
                     num_classes=num_classes, num_layers=num_layers, dropout=dropout)
    elif mtype == "tcn":
        model = TCN(input_size=input_size, hidden_size=hidden_size,
                     num_classes=num_classes, num_layers=num_layers, dropout=dropout)
    else:
        model = BiLSTM(input_size=input_size, hidden_size=hidden_size,
                        num_classes=num_classes, num_layers=num_layers, dropout=dropout)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(
        f"[inference] Loaded {mtype} checkpoint: "
        f"{num_classes} classes, {input_size} features.",
        flush=True,
    )
    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "config": cfg,
    }


def _auto_discover_checkpoint(task: str = "activity_recognition") -> Path:
    """Scan ``outputs/`` for the most recent checkpoint matching a task."""
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError("No outputs/ directory found. Train a model first.")

    candidates = sorted(
        outputs_dir.rglob("best_model.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for p in candidates:
        try:
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            t_name = cfg.get("task", {}).get("name", "")
            if t_name == task:
                print(f"[inference] Auto-discovered checkpoint: {p}", flush=True)
                return p
        except Exception:
            continue

    raise FileNotFoundError(
        f"No checkpoint found for task '{task}' in {outputs_dir}."
    )


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────

def predict_from_video(
    video_path: str,
    checkpoint_path: str | None = None,
    videopose3d_checkpoint: str | None = None,
    task: str = "activity_recognition",
    window_size: int = 60,
    stride: int = 30,
    frame_skip_step: int = 2,
    pad_target_len: int = 60,
    mediapipe_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> dict:
    """
    Full pipeline: Video → 2D → 3D → Features → Windows → BiLSTM → Predictions.

    Parameters
    ----------
    video_path : str
        Path to input video file.
    checkpoint_path : str, optional
        Path to ``best_model.pt``. Auto-discovers if None.
    videopose3d_checkpoint : str, optional
        Path to VideoPose3D ``.bin`` weights. Auto-downloads if None.
    task : str
        Task name for auto-discovery (default ``"activity_recognition"``).
    window_size : int
        Sliding window size (frames).
    stride : int
        Sliding window stride.
    frame_skip_step : int
        Sub-sample every N-th frame within each window.
    pad_target_len : int
        Pad / truncate windows to this length.
    mediapipe_complexity : int
        MediaPipe model complexity (0/1/2).
    min_detection_confidence : float
        MediaPipe detection confidence threshold.
    min_tracking_confidence : float
        MediaPipe tracking confidence threshold.

    Returns
    -------
    dict
        {
            "predictions": [
                {"window_idx": int, "label": str, "confidence": float,
                 "probabilities": {label: prob, ...}},
                ...
            ],
            "summary": {label: count, ...},
            "metadata": {"fps": float, "total_frames": int, ...}
        }
    """
    # ── Stage 1: Video → 2D ─────────────────────────────────────────
    print("=" * 60)
    print("STAGE 1: Extracting 2D poses with MediaPipe")
    print("=" * 60)
    keypoints_2d, video_meta = extract_2d_poses(
        video_path,
        model_complexity=mediapipe_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    if video_meta["detected_frames"] == 0:
        return {
            "predictions": [],
            "summary": {},
            "metadata": video_meta,
            "error": "No poses detected in video.",
        }

    # ── Stage 2: 2D → 3D ────────────────────────────────────────────
    print("=" * 60)
    print("STAGE 2: Lifting to 3D with VideoPose3D")
    print("=" * 60)
    joints_3d = lift_to_3d(
        keypoints_2d,
        checkpoint_path=videopose3d_checkpoint,
    )

    # ── Stage 3: Map H3.6M → Xsens DataFrame ────────────────────────
    print("=" * 60)
    print("STAGE 3: Mapping joints to Xsens segment format")
    print("=" * 60)
    df = map_h36m_to_xsens(joints_3d)

    # Inject required metadata columns for the preprocessing pipeline
    video_name = Path(video_path).stem
    df["video_id"] = video_name
    df["sensor_type"] = "Segment Position"
    df["Label"] = 0  # placeholder — required by create_windows but not used for inference

    # ── Stage 4: Merge Sensors (single sensor = just adds _pos suffix) ──
    print("=" * 60)
    print("STAGE 4: Feature engineering")
    print("=" * 60)
    sensor_types = ["Segment Position"]
    merged_df, base_feature_cols = merge_sensors(df, sensor_types)

    # ── Stage 5: Engineer Features ───────────────────────────────────
    merged_df, feature_cols = engineer_features(merged_df, base_feature_cols)
    print(f"  Engineered {len(feature_cols)} feature columns.", flush=True)

    # ── Stage 6: Create Windows ──────────────────────────────────────
    print("=" * 60)
    print("STAGE 5: Creating sliding windows")
    print("=" * 60)
    X_windows, y_windows, window_meta_df = create_windows(
        merged_df,
        feature_cols,
        window_size,
        stride,
        target_col="Label",
        frame_skip_step=frame_skip_step,
    )

    if len(X_windows) == 0:
        return {
            "predictions": [],
            "summary": {},
            "metadata": video_meta,
            "error": "Video too short to create any windows.",
        }

    X_all = pad_windows_to_60(X_windows, target_len=pad_target_len).astype(np.float32)
    X_all = clean_features(X_all)

    print(f"  Windows created: {X_all.shape[0]}, shape per window: {X_all.shape[1:]}", flush=True)

    # ── Stage 7: Load BiLSTM and predict ─────────────────────────────
    print("=" * 60)
    print("STAGE 6: Running BiLSTM inference")
    print("=" * 60)

    if checkpoint_path is None:
        checkpoint_path = str(_auto_discover_checkpoint(task))

    bundle = _load_bilstm_checkpoint(checkpoint_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]

    # Validate feature count
    expected_features = scaler.n_features_in_
    actual_features = X_all.shape[2]
    if actual_features != expected_features:
        return {
            "predictions": [],
            "summary": {},
            "metadata": video_meta,
            "error": (
                f"Feature mismatch: model expects {expected_features} features "
                f"but pipeline produced {actual_features}. "
                f"Ensure the model was trained with sensor_types=['Segment Position']."
            ),
        }

    # Scale
    B, T, Feat = X_all.shape
    X_scaled = scaler.transform(X_all.reshape(-1, Feat)).reshape(B, T, Feat)

    # Predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    predictions = []
    BATCH_SIZE = 64

    with torch.no_grad():
        for start in range(0, B, BATCH_SIZE):
            end = min(start + BATCH_SIZE, B)
            batch = tensor[start:end]
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            for i in range(probs.shape[0]):
                pred_idx = int(np.argmax(probs[i]))
                pred_label = label_encoder.inverse_transform([pred_idx])[0]

                prob_dict = {
                    str(cls): round(float(p), 4)
                    for cls, p in zip(label_encoder.classes_, probs[i])
                }

                predictions.append({
                    "window_idx": start + i,
                    "label": str(pred_label),
                    "confidence": round(float(probs[i][pred_idx]), 4),
                    "probabilities": prob_dict,
                })

    # Build summary
    summary: dict[str, int] = {}
    for pred in predictions:
        lbl = pred["label"]
        summary[lbl] = summary.get(lbl, 0) + 1

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total windows: {len(predictions)}")
    for lbl, count in sorted(summary.items()):
        print(f"  Class '{lbl}': {count} windows ({count / len(predictions) * 100:.1f}%)")

    return {
        "predictions": predictions,
        "summary": summary,
        "metadata": video_meta,
    }


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run video → activity prediction pipeline."
    )
    parser.add_argument("video", type=str, help="Path to input video file.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to best_model.pt. Auto-discovers if omitted.")
    parser.add_argument("--videopose3d-checkpoint", type=str, default=None,
                        help="Path to VideoPose3D .bin weights. Auto-downloads if omitted.")
    parser.add_argument("--task", type=str, default="activity_recognition",
                        help="Task name for auto-discovery.")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--frame-skip-step", type=int, default=2)
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file.")
    args = parser.parse_args()

    result = predict_from_video(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        videopose3d_checkpoint=args.videopose3d_checkpoint,
        task=args.task,
        window_size=args.window_size,
        stride=args.stride,
        frame_skip_step=args.frame_skip_step,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output_json}")
