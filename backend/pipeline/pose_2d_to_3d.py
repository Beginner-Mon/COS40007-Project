"""
Stage 2 — Lift 2D MediaPipe poses to 3D using a pretrained VideoPose3D model.

Pipeline:
    1. Remap MediaPipe 33-landmark format → Human3.6M 17-joint format.
    2. Normalise 2D keypoints (subtract hip, optionally scale).
    3. Load pretrained VideoPose3D temporal convolution model.
    4. Run inference to produce 3D joint positions.
"""

import numpy as np
import torch
from pathlib import Path

from .videopose3d_model import TemporalModel
from .download_checkpoint import download_checkpoint, CHECKPOINT_FILE

# ─────────────────────────────────────────────────────────────────────
# MediaPipe 33 → Human3.6M 17 joint mapping
# ─────────────────────────────────────────────────────────────────────
# Human3.6M joint order:
#  0: Hip (pelvis center)      8:  Thorax
#  1: Right Hip                9:  Neck / Nose
#  2: Right Knee               10: Head
#  3: Right Ankle              11: Left Shoulder
#  4: Left Hip                 12: Left Elbow
#  5: Left Knee                13: Left Wrist
#  6: Left Ankle               14: Right Shoulder
#  7: Spine                    15: Right Elbow
#                              16: Right Wrist
#
# MediaPipe landmark indices used:
#  0:  Nose                    23: Left Hip       27: Left Ankle
#  11: Left Shoulder           24: Right Hip      28: Right Ankle
#  12: Right Shoulder          25: Left Knee
#  13: Left Elbow              26: Right Knee
#  14: Right Elbow
#  15: Left Wrist
#  16: Right Wrist

# Joints that are direct 1:1 mappings (H3.6M index → MediaPipe index)
_DIRECT_MAP = {
    1: 24,   # Right Hip
    2: 26,   # Right Knee
    3: 28,   # Right Ankle
    4: 23,   # Left Hip
    5: 25,   # Left Knee
    6: 27,   # Left Ankle
    9: 0,    # Neck / Nose  (approx — MP has no true neck, nose is closest)
    10: 0,   # Head         (approx — using nose)
    11: 11,  # Left Shoulder
    12: 13,  # Left Elbow
    13: 15,  # Left Wrist
    14: 12,  # Right Shoulder
    15: 14,  # Right Elbow
    16: 16,  # Right Wrist
}

# Joints that are midpoints of two MediaPipe landmarks
_MIDPOINT_MAP = {
    0: (23, 24),  # Hip center = midpoint(left hip, right hip)
    7: (23, 11),  # Spine = midpoint(left hip, left shoulder)  — rough approx
    8: (11, 12),  # Thorax = midpoint(left shoulder, right shoulder)
}


def mediapipe_to_h36m(keypoints_mp: np.ndarray) -> np.ndarray:
    """
    Convert MediaPipe 33-landmark keypoints to Human3.6M 17-joint format.

    Parameters
    ----------
    keypoints_mp : np.ndarray
        Shape ``(num_frames, 33, 4)`` — MediaPipe landmarks (x, y, z, visibility).

    Returns
    -------
    np.ndarray
        Shape ``(num_frames, 17, 2)`` — 2D keypoints (x, y) in H3.6M joint order.
    """
    N = keypoints_mp.shape[0]
    h36m = np.zeros((N, 17, 2), dtype=np.float32)

    for h36m_idx, mp_idx in _DIRECT_MAP.items():
        h36m[:, h36m_idx, :] = keypoints_mp[:, mp_idx, :2]  # take x, y only

    for h36m_idx, (mp_a, mp_b) in _MIDPOINT_MAP.items():
        h36m[:, h36m_idx, :] = (
            keypoints_mp[:, mp_a, :2] + keypoints_mp[:, mp_b, :2]
        ) / 2.0

    return h36m


def _normalise_2d(keypoints_2d: np.ndarray) -> np.ndarray:
    """
    Normalise 2D keypoints: subtract hip (joint 0) and scale by bbox width.

    Parameters
    ----------
    keypoints_2d : np.ndarray
        Shape ``(N, 17, 2)``.

    Returns
    -------
    np.ndarray
        Normalised keypoints, same shape.
    """
    kp = keypoints_2d.copy()

    # Subtract hip center
    hip = kp[:, 0:1, :]  # (N, 1, 2)
    kp = kp - hip

    # Scale by approximate body height (distance from hip to head)
    head = kp[:, 10, :]   # (N, 2)
    scale = np.linalg.norm(head, axis=-1, keepdims=True)  # (N, 1)
    scale = np.clip(scale, a_min=1e-6, a_max=None)
    kp = kp / scale[:, np.newaxis, :]

    return kp


def _interpolate_nan_frames(keypoints: np.ndarray) -> np.ndarray:
    """
    Fill NaN frames via linear interpolation along the time axis.
    Leading/trailing NaN blocks are forward/backward filled.
    """
    N, J, C = keypoints.shape
    result = keypoints.copy()

    for j in range(J):
        for c in range(C):
            col = result[:, j, c]
            nans = np.isnan(col)
            if nans.all():
                col[:] = 0.0
                continue
            if nans.any():
                valid = ~nans
                indices = np.arange(N)
                col[nans] = np.interp(indices[nans], indices[valid], col[valid])
            result[:, j, c] = col

    return result


def load_videopose3d_model(
    checkpoint_path: str | Path | None = None,
    num_joints: int = 17,
    filter_widths: list[int] | None = None,
    channels: int = 1024,
) -> tuple[TemporalModel, int]:
    """
    Load the pretrained VideoPose3D model.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to ``pretrained_h36m_detectron_coco.bin``.
        If None, auto-downloads to the default location.
    num_joints : int
        Number of joints (17 for H3.6M).
    filter_widths : list[int], optional
        Kernel widths. Default ``[3, 3, 3, 3, 3]``.
    channels : int
        Hidden channels. Default 1024.

    Returns
    -------
    model : TemporalModel
        Loaded model in eval mode.
    receptive_field : int
        Number of input frames the model needs.
    """
    if filter_widths is None:
        filter_widths = [3, 3, 3, 3, 3]

    # Resolve checkpoint
    if checkpoint_path is None:
        if not CHECKPOINT_FILE.exists():
            print("[pose_2d_to_3d] Checkpoint not found — downloading ...", flush=True)
            download_checkpoint()
        checkpoint_path = CHECKPOINT_FILE
    else:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Build model
    model = TemporalModel(
        num_joints_in=num_joints,
        in_features=2,
        num_joints_out=num_joints,
        filter_widths=filter_widths,
        causal=False,
        dropout=0.25,
        channels=channels,
    )

    # Load weights
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_pos"])
    model.eval()

    receptive_field = model.receptive_field()
    print(
        f"[pose_2d_to_3d] VideoPose3D loaded. "
        f"Receptive field = {receptive_field} frames.",
        flush=True,
    )
    return model, receptive_field


def lift_to_3d(
    keypoints_mp: np.ndarray,
    checkpoint_path: str | Path | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Full 2D→3D lifting pipeline.

    Parameters
    ----------
    keypoints_mp : np.ndarray
        Raw MediaPipe output, shape ``(num_frames, 33, 4)``.
    checkpoint_path : str or Path, optional
        Path to VideoPose3D checkpoint. Auto-downloads if None.
    batch_size : int
        Batch size for model inference.

    Returns
    -------
    np.ndarray
        3D joint positions of shape ``(num_frames, 17, 3)``.
    """
    # Step 1: MediaPipe → H3.6M 2D
    kp_2d = mediapipe_to_h36m(keypoints_mp)  # (N, 17, 2)

    # Step 2: Interpolate NaN frames (where MediaPipe failed to detect)
    kp_2d = _interpolate_nan_frames(kp_2d)

    # Step 3: Normalise
    kp_2d_norm = _normalise_2d(kp_2d)

    # Step 4: Load model
    model, receptive_field = load_videopose3d_model(checkpoint_path)

    # Step 5: Prepare input — pad sequence so the model's temporal shrinkage
    # still produces N output frames (model shrinks by receptive_field - 1).
    N = kp_2d_norm.shape[0]
    pad = receptive_field // 2  # padding on each side

    # Replicate-pad along time axis: (N, 17, 2) → (N + 2*pad, 17, 2)
    kp_padded = np.pad(kp_2d_norm, ((pad, pad), (0, 0), (0, 0)), mode="edge")

    # Model expects 4D: (batch, frames, joints, features)
    input_tensor = torch.from_numpy(kp_padded).float().unsqueeze(0)  # (1, N+2*pad, 17, 2)

    # Step 6: Run inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        output = model(input_tensor.to(device))  # (1, N, 17, 3)

    joints_3d = output.squeeze(0).cpu().numpy()  # (N, 17, 3)

    # Trim to original length (safety — should already be N)
    joints_3d = joints_3d[:N]

    print(f"[pose_2d_to_3d] 3D lifting complete. Output shape: {joints_3d.shape}", flush=True)
    return joints_3d
