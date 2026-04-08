"""
Stage 3 — Map Human3.6M 17-joint skeleton → Xsens 23-segment DataFrame.

Produces a DataFrame whose columns exactly match the ``FEATURE_COLS`` list
in ``data.preprocessing``, so the downstream feature-engineering and windowing
code can be reused without modification.
"""

import numpy as np
import pandas as pd

from data.preprocessing import FEATURE_COLS

# ─────────────────────────────────────────────────────────────────────
# H3.6M joint indices (same order as VideoPose3D output)
# ─────────────────────────────────────────────────────────────────────
H36M_HIP = 0
H36M_RHIP = 1
H36M_RKNEE = 2
H36M_RANKLE = 3
H36M_LHIP = 4
H36M_LKNEE = 5
H36M_LANKLE = 6
H36M_SPINE = 7
H36M_THORAX = 8
H36M_NECK = 9
H36M_HEAD = 10
H36M_LSHOULDER = 11
H36M_LELBOW = 12
H36M_LWRIST = 13
H36M_RSHOULDER = 14
H36M_RELBOW = 15
H36M_RWRIST = 16

# ─────────────────────────────────────────────────────────────────────
# Xsens segment names (order matches FEATURE_COLS grouping)
# ─────────────────────────────────────────────────────────────────────
XSENS_SEGMENTS = [
    "L5", "L3", "T12", "T8", "Neck", "Head",
    "Right Shoulder", "Right Upper Arm", "Right Forearm", "Right Hand",
    "Left Shoulder", "Left Upper Arm", "Left Forearm", "Left Hand",
    "Right Upper Leg", "Right Lower Leg", "Right Foot", "Right Toe",
    "Left Upper Leg", "Left Lower Leg", "Left Foot", "Left Toe",
]

# ─────────────────────────────────────────────────────────────────────
# Mapping rules: Xsens segment → how to derive from H3.6M joints
#   "direct": (h36m_index,)
#   "midpoint": (h36m_a, h36m_b)
# ─────────────────────────────────────────────────────────────────────
_SEGMENT_RULES: dict[str, tuple[str, tuple[int, ...]]] = {
    # Spine chain
    "L5":               ("direct",   (H36M_HIP,)),
    "L3":               ("midpoint", (H36M_HIP, H36M_SPINE)),
    "T12":              ("direct",   (H36M_SPINE,)),
    "T8":               ("direct",   (H36M_THORAX,)),
    "Neck":             ("direct",   (H36M_NECK,)),
    "Head":             ("direct",   (H36M_HEAD,)),
    # Right arm
    "Right Shoulder":   ("direct",   (H36M_RSHOULDER,)),
    "Right Upper Arm":  ("midpoint", (H36M_RSHOULDER, H36M_RELBOW)),
    "Right Forearm":    ("direct",   (H36M_RELBOW,)),
    "Right Hand":       ("direct",   (H36M_RWRIST,)),
    # Left arm
    "Left Shoulder":    ("direct",   (H36M_LSHOULDER,)),
    "Left Upper Arm":   ("midpoint", (H36M_LSHOULDER, H36M_LELBOW)),
    "Left Forearm":     ("direct",   (H36M_LELBOW,)),
    "Left Hand":        ("direct",   (H36M_LWRIST,)),
    # Right leg
    "Right Upper Leg":  ("direct",   (H36M_RHIP,)),
    "Right Lower Leg":  ("direct",   (H36M_RKNEE,)),
    "Right Foot":       ("direct",   (H36M_RANKLE,)),
    "Right Toe":        ("direct",   (H36M_RANKLE,)),   # duplicated — no toe in H3.6M
    # Left leg
    "Left Upper Leg":   ("direct",   (H36M_LHIP,)),
    "Left Lower Leg":   ("direct",   (H36M_LKNEE,)),
    "Left Foot":        ("direct",   (H36M_LANKLE,)),
    "Left Toe":         ("direct",   (H36M_LANKLE,)),    # duplicated — no toe in H3.6M
}


def map_h36m_to_xsens(joints_3d: np.ndarray) -> pd.DataFrame:
    """
    Convert VideoPose3D output to a DataFrame matching ``FEATURE_COLS``.

    Parameters
    ----------
    joints_3d : np.ndarray
        Shape ``(num_frames, 17, 3)`` — 3D joint positions from VideoPose3D.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Frame`` + all 69 ``FEATURE_COLS``.
        One row per frame.
    """
    N = joints_3d.shape[0]
    data: dict[str, np.ndarray] = {}

    data["Frame"] = np.arange(1, N + 1)

    for segment_name in XSENS_SEGMENTS:
        rule_type, indices = _SEGMENT_RULES[segment_name]

        if rule_type == "direct":
            joint_idx = indices[0]
            pos = joints_3d[:, joint_idx, :]  # (N, 3)
        elif rule_type == "midpoint":
            idx_a, idx_b = indices
            pos = (joints_3d[:, idx_a, :] + joints_3d[:, idx_b, :]) / 2.0
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")

        data[f"{segment_name} x"] = pos[:, 0]
        data[f"{segment_name} y"] = pos[:, 1]
        data[f"{segment_name} z"] = pos[:, 2]

    df = pd.DataFrame(data)

    # Validate: ensure all FEATURE_COLS are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Joint mapping produced incomplete columns. Missing: {missing}"
        )

    print(
        f"[joint_mapper] Mapped {N} frames → DataFrame with "
        f"{len(FEATURE_COLS)} feature columns.",
        flush=True,
    )
    return df
