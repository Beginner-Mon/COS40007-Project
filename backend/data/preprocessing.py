import numpy as np
from sklearn.preprocessing import StandardScaler

# Explicit feature list to avoid column drift across files.
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

def create_windows(df, feature_cols, window_size, stride=1, label_col="Label"):
    X, y = [], []

    for _, video_df in df.groupby("video_id"):
        video_df = video_df.sort_values("Frame")

        Xv = video_df[feature_cols].values
        yv = video_df[label_col].values

        for i in range(0, len(Xv) - window_size + 1, stride):
            X.append(Xv[i:i + window_size])
            y.append(yv[i + window_size - 1])

    return np.array(X), np.array(y)

def clean_features(X):
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def normalize_features(X):
    B, T, F = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, F)).reshape(B, T, F)
    return X, scaler
