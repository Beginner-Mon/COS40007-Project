import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def derive_sharpness_class(df):
    """Derive 'sharpness_class' column from 'knife_sharpness_score'.
    Bins: sharp (>=85), medium (70-84), blunt (<70).
    """
    if "knife_sharpness_score" not in df.columns:
        raise ValueError("Cannot derive sharpness_class: 'knife_sharpness_score' column not found.")
    bins = [-float("inf"), 70, 85, float("inf")]
    labels = ["blunt", "medium", "sharp"]
    df["sharpness_class"] = pd.cut(
        df["knife_sharpness_score"], bins=bins, labels=labels, right=False
    ).astype(str)
    return df


LABEL_MAPPING = {
    '0- Idle': 0,
    '0 - Idle': 0,
    '1- Walking': 1,
    '1 - Walking': 1,
    '2- Steeling': 2,
    '2 - Steeling': 2,
    '3- Reaching': 3,
    '3 - Reaching': 3,
    '4- Cutting': 4,
    '4 - Cutting': 4,
    '4 - Cutting (Big Pieces)': 4,
    '4 - Cutting (Big Piece)': 4,
    '4- Cutting (big piece)': 4,
    '4 - Cutting (offloading bone from Carcass)': 4,
    '5- Dropping': 8,
    '5 - Dropping': 8,
    '5 - Slicing': 5,
    '6 - Pulling': 6,
    '7 - Placing/ Manipulating': 7,
    '8 - Dropping': 8,
}

def clean_labels(df, target_col="Label", activity_col="activity_type"):
    """Map string labels to consistent integers and apply Boning 5->8 overrides."""
    if target_col not in df.columns:
        return df

    out_df = df.copy()

    # 1. Base mapping to integers
    out_df[target_col] = out_df[target_col].replace(LABEL_MAPPING)
    out_df[target_col] = pd.to_numeric(out_df[target_col], errors='coerce')
    out_df = out_df.dropna(subset=[target_col]).copy()
    out_df[target_col] = out_df[target_col].astype('int64')

    # 2. Apply boning override: Label 5 -> 8
    if activity_col in out_df.columns:
        activity_series = out_df[activity_col].astype(str).str.strip().str.lower()
        boning_mask = activity_series.eq('boning')
        override_mask = boning_mask & out_df[target_col].eq(5)
        changed = int(override_mask.sum())
        
        if changed > 0:
            out_df.loc[override_mask, target_col] = 8
            print(f"[phase=data] Auto-corrected {changed} 'boning' labels from 5 to 8.", flush=True)

    return out_df


FEATURE_COLS = [
    "L5 x", "L5 y", "L5 z",
    "L3 x", "L3 y", "L3 z",
    "T12 x", "T12 y", "T12 z",
    "T8 x", "T8 y", "T8 z",
    "Neck x", "Neck y", "Neck z",
    "Head x", "Head y", "Head z",
    "Right Shoulder x", "Right Shoulder y", "Right Shoulder z",
    "Right Upper Arm x", "Right Upper Arm y", "Right Upper Arm z",
    "Right Forearm x", "Right Forearm y", "Right Forearm z",
    "Right Hand x", "Right Hand y", "Right Hand z",
    "Left Shoulder x", "Left Shoulder y", "Left Shoulder z",
    "Left Upper Arm x", "Left Upper Arm y", "Left Upper Arm z",
    "Left Forearm x", "Left Forearm y", "Left Forearm z",
    "Left Hand x", "Left Hand y", "Left Hand z",
    "Right Upper Leg x", "Right Upper Leg y", "Right Upper Leg z",
    "Right Lower Leg x", "Right Lower Leg y", "Right Lower Leg z",
    "Right Foot x", "Right Foot y", "Right Foot z",
    "Right Toe x", "Right Toe y", "Right Toe z",
    "Left Upper Leg x", "Left Upper Leg y", "Left Upper Leg z",
    "Left Lower Leg x", "Left Lower Leg y", "Left Lower Leg z",
    "Left Foot x", "Left Foot y", "Left Foot z",
    "Left Toe x", "Left Toe y", "Left Toe z",
]

def get_feature_columns(df):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing feature columns: {missing_str}")
    return FEATURE_COLS

def merge_velocity_and_acceleration(df):
    vel_df = df[df["sensor_type"] == "Segment Velocity"].copy()
    acc_df = df[df["sensor_type"] == "Segment Acceleration"].copy()

    if vel_df.empty or acc_df.empty:
        raise ValueError("Both Segment Velocity and Segment Acceleration rows are required.")

    base_feature_cols = get_feature_columns(vel_df)

    id_cols = [
        c for c in ["video_id", "Frame", "Label", "person_id", "activity_type", "knife_sharpness_score", "sharpness_class"]
        if c in vel_df.columns
    ]

    vel_df = vel_df[id_cols + base_feature_cols].rename(
        columns={c: f"{c}_vel" for c in base_feature_cols}
    )
    acc_df = acc_df[["video_id", "Frame"] + base_feature_cols].rename(
        columns={c: f"{c}_acc" for c in base_feature_cols}
    )

    merged_df = vel_df.merge(
        acc_df,
        on=["video_id", "Frame"],
        how="inner",
        validate="one_to_one",
    )
    return merged_df, base_feature_cols

def engineer_features(combined_df, base_feature_cols):
    segment_names = []
    for col in base_feature_cols:
        segment_name, axis = col.rsplit(" ", 1)
        if axis == "x":
            segment_names.append(segment_name)

    segment_names = list(dict.fromkeys(segment_names))
    engineered_cols = {}

    velocity_mag_feature_cols = []
    acceleration_mag_feature_cols = []
    for seg in segment_names:
        vx_col, vy_col, vz_col = f"{seg} x_vel", f"{seg} y_vel", f"{seg} z_vel"
        ax_col, ay_col, az_col = f"{seg} x_acc", f"{seg} y_acc", f"{seg} z_acc"

        vel_mag_col = f"{seg}_vel_mag"
        acc_mag_col = f"{seg}_acc_mag"

        engineered_cols[vel_mag_col] = np.sqrt(
            combined_df[vx_col] ** 2 + combined_df[vy_col] ** 2 + combined_df[vz_col] ** 2
        )
        engineered_cols[acc_mag_col] = np.sqrt(
            combined_df[ax_col] ** 2 + combined_df[ay_col] ** 2 + combined_df[az_col] ** 2
        )
        velocity_mag_feature_cols.append(vel_mag_col)
        acceleration_mag_feature_cols.append(acc_mag_col)

    energy_terms = []
    for seg in segment_names:
        energy_terms.extend([
            combined_df[f"{seg} x_vel"] ** 2,
            combined_df[f"{seg} y_vel"] ** 2,
            combined_df[f"{seg} z_vel"] ** 2,
            combined_df[f"{seg} x_acc"] ** 2,
            combined_df[f"{seg} y_acc"] ** 2,
            combined_df[f"{seg} z_acc"] ** 2,
        ])
    engineered_cols["total_body_energy"] = np.sum(np.column_stack(energy_terms), axis=1)

    arm_joints = [seg for seg in segment_names if any(k in seg for k in ["Shoulder", "Upper Arm", "Forearm", "Hand"])]
    leg_joints = [seg for seg in segment_names if any(k in seg for k in ["Upper Leg", "Lower Leg", "Foot", "Toe"])]
    arm_leg_joints = list(dict.fromkeys(arm_joints + leg_joints))

    spine_head_joints = [seg for seg in segment_names if seg in ["L5", "L3", "T12", "T8", "Neck", "Head"]]
    upper_body_joints = list(dict.fromkeys(spine_head_joints + arm_joints))
    lower_body_joints = leg_joints
    left_body_joints = [seg for seg in segment_names if seg.startswith("Left ")]
    right_body_joints = [seg for seg in segment_names if seg.startswith("Right ")]

    pair_suffixes = [("x", "y", "xy"), ("x", "z", "xz"), ("y", "z", "yz")]
    pair_feature_cols = []
    for seg in arm_leg_joints:
        for a1, a2, pair_name in pair_suffixes:
            vel_pair_col = f"{seg}_vel_{pair_name}"
            acc_pair_col = f"{seg}_acc_{pair_name}"
            engineered_cols[vel_pair_col] = np.sqrt(combined_df[f"{seg} {a1}_vel"] ** 2 + combined_df[f"{seg} {a2}_vel"] ** 2)
            engineered_cols[acc_pair_col] = np.sqrt(combined_df[f"{seg} {a1}_acc"] ** 2 + combined_df[f"{seg} {a2}_acc"] ** 2)
            pair_feature_cols.extend([vel_pair_col, acc_pair_col])

    def group_energy(df, joints):
        if len(joints) == 0:
            return pd.Series(np.zeros(len(df)), index=df.index)
        terms = []
        for seg in joints:
            terms.extend([
                df[f"{seg} x_vel"] ** 2, df[f"{seg} y_vel"] ** 2, df[f"{seg} z_vel"] ** 2,
                df[f"{seg} x_acc"] ** 2, df[f"{seg} y_acc"] ** 2, df[f"{seg} z_acc"] ** 2,
            ])
        return np.sum(np.column_stack(terms), axis=1)

    upper_energy = group_energy(combined_df, upper_body_joints)
    lower_energy = group_energy(combined_df, lower_body_joints)
    left_energy = group_energy(combined_df, left_body_joints)
    right_energy = group_energy(combined_df, right_body_joints)

    engineered_cols["upper_body_energy"] = upper_energy
    engineered_cols["lower_body_energy"] = lower_energy
    engineered_cols["left_body_energy"] = left_energy
    engineered_cols["right_body_energy"] = right_energy

    eps = 1e-8
    ul_denom = upper_energy + lower_energy + eps
    lr_denom = left_energy + right_energy + eps

    engineered_cols["upper_energy_ratio"] = upper_energy / ul_denom
    engineered_cols["lower_energy_ratio"] = lower_energy / ul_denom
    engineered_cols["left_energy_ratio"] = left_energy / lr_denom
    engineered_cols["right_energy_ratio"] = right_energy / lr_denom

    energy_distribution_feature_cols = [
        "upper_energy_ratio", "lower_energy_ratio",
        "left_energy_ratio", "right_energy_ratio",
    ]

    combined_df = pd.concat([combined_df, pd.DataFrame(engineered_cols)], axis=1)

    feature_cols = (
        velocity_mag_feature_cols
        + acceleration_mag_feature_cols
        + ["total_body_energy"]
        + pair_feature_cols
        + energy_distribution_feature_cols
    )
    return combined_df, feature_cols

def create_windows(
    df,
    feature_cols,
    window_size,
    stride,
    target_col="Label",
    frame_skip_step=2,
    label_specific_rules=None,
):
    if len(df) == 0:
        raise ValueError("Cannot create windows from an empty dataframe.")

    default_window_size = int(window_size)
    default_stride = int(stride)
    default_frame_skip_step = int(frame_skip_step)

    if default_window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {default_window_size}")
    if default_stride <= 0:
        raise ValueError(f"stride must be > 0, got {default_stride}")
    if default_frame_skip_step <= 0:
        raise ValueError(f"frame_skip_step must be > 0, got {default_frame_skip_step}")

    label_specific_rules = label_specific_rules or {}

    def _resolve_window_params(target_value):
        params = {
            "window_size": default_window_size,
            "window_stride": default_stride,
            "frame_skip_step": default_frame_skip_step,
        }

        target_key = str(target_value).strip().lower()
        selected_rule = None
        for rule_key, rule_value in label_specific_rules.items():
            if str(rule_key).strip().lower() == target_key:
                selected_rule = rule_value
                break

        if isinstance(selected_rule, dict):
            if "window_size" in selected_rule:
                params["window_size"] = int(selected_rule["window_size"])
            if "window_stride" in selected_rule:
                params["window_stride"] = int(selected_rule["window_stride"])
            elif "stride" in selected_rule:
                params["window_stride"] = int(selected_rule["stride"])
            if "frame_skip_step" in selected_rule:
                params["frame_skip_step"] = int(selected_rule["frame_skip_step"])

        if params["window_size"] <= 0:
            raise ValueError(
                f"window_size must be > 0 for target '{target_value}', got {params['window_size']}"
            )
        if params["window_stride"] <= 0:
            raise ValueError(
                f"window_stride must be > 0 for target '{target_value}', got {params['window_stride']}"
            )
        if params["frame_skip_step"] <= 0:
            raise ValueError(
                f"frame_skip_step must be > 0 for target '{target_value}', got {params['frame_skip_step']}"
            )

        return params

    df = df.sort_values(["video_id", "Frame"]).reset_index(drop=True)
    runs = []
    
    start_idx = 0
    current_video = df.loc[0, "video_id"]
    current_target = df.loc[0, target_col]
    
    for i in range(1, len(df)):
        video_i = df.loc[i, "video_id"]
        target_i = df.loc[i, target_col]
        
        # Enforce temporal continuity
        frame_i = df.loc[i, "Frame"]
        frame_prev = df.loc[i - 1, "Frame"]
        is_continuous = (frame_i - frame_prev) == 1
        
        if (video_i != current_video) or (target_i != current_target) or not is_continuous:
            end_idx = i - 1
            runs.append({
                "video_id": current_video, "target": current_target,
                "start_idx": start_idx, "end_idx": end_idx
            })
            start_idx = i
            current_video = video_i
            current_target = target_i

    end_idx = len(df) - 1
    runs.append({
        "video_id": current_video, "target": current_target,
        "start_idx": start_idx, "end_idx": end_idx
    })

    X_windows = []
    y_windows = []
    window_meta = []
    
    for run_idx, run in enumerate(runs):
        params = _resolve_window_params(run["target"])
        run_window_size = params["window_size"]
        run_window_stride = params["window_stride"]
        run_frame_skip_step = params["frame_skip_step"]

        run_indices = np.arange(run["start_idx"], run["end_idx"] + 1)
        sampled_indices = run_indices[::run_frame_skip_step]
        if len(sampled_indices) == 0:
            continue
        
        if len(sampled_indices) <= run_window_size:
            idx_chunk = sampled_indices
            X_windows.append(df.iloc[idx_chunk][feature_cols].to_numpy())
            y_windows.append(run["target"])
            window_meta.append(
                {
                    "video_id": run["video_id"],
                    "target": run["target"],
                    "run_idx": run_idx,
                    "window_len": int(len(idx_chunk)),
                    "window_size": run_window_size,
                    "window_stride": run_window_stride,
                    "frame_skip_step": run_frame_skip_step,
                }
            )
            continue

        starts = list(range(0, len(sampled_indices) - run_window_size + 1, run_window_stride))
        for start in starts:
            idx_chunk = sampled_indices[start:start + run_window_size]
            X_windows.append(df.iloc[idx_chunk][feature_cols].to_numpy())
            y_windows.append(run["target"])
            window_meta.append(
                {
                    "video_id": run["video_id"],
                    "target": run["target"],
                    "run_idx": run_idx,
                    "window_len": int(len(idx_chunk)),
                    "window_size": run_window_size,
                    "window_stride": run_window_stride,
                    "frame_skip_step": run_frame_skip_step,
                }
            )

        last_covered = starts[-1] + run_window_size
        if last_covered < len(sampled_indices):
            idx_chunk = sampled_indices[last_covered:]
            X_windows.append(df.iloc[idx_chunk][feature_cols].to_numpy())
            y_windows.append(run["target"])
            window_meta.append(
                {
                    "video_id": run["video_id"],
                    "target": run["target"],
                    "run_idx": run_idx,
                    "window_len": int(len(idx_chunk)),
                    "window_size": run_window_size,
                    "window_stride": run_window_stride,
                    "frame_skip_step": run_frame_skip_step,
                }
            )
            
    return X_windows, y_windows, pd.DataFrame(window_meta)

def pad_windows_to_60(window_list, target_len=60):
    padded = []
    for arr in window_list:
        if arr.shape[0] == target_len:
            padded.append(arr)
        elif arr.shape[0] < target_len:
            pad_rows = target_len - arr.shape[0]
            pad_block = np.zeros((pad_rows, arr.shape[1]), dtype=arr.dtype)
            padded.append(np.vstack([arr, pad_block]))
        else:
            padded.append(arr[:target_len])
    return np.stack(padded, axis=0)

def clean_features(X):
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
