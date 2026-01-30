import numpy as np
from sklearn.preprocessing import StandardScaler

NON_FEATURE_COLS = [
    "Frame", "Label", "person_id", "activity_type",
    "knife_sharpness_score", "sensor_type", "video_id"
]

def get_feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]

def create_windows(df, feature_cols, window_size):
    X, y = [], []

    for _, video_df in df.groupby("video_id"):
        video_df = video_df.sort_values("Frame")

        Xv = video_df[feature_cols].values
        yv = video_df["activity_type"].values

        for i in range(len(Xv) - window_size + 1):
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
