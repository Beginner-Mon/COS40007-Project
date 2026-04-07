from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))

PROJECT_ROOT
---
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.seed import set_seed
from utils.device import get_device
from data.preprocessing import (
    create_windows,
    clean_features,
    get_feature_columns,
 )
from data.motion_dataset import MotionDataset

# Config (no Hydra)
seed = 42
device = get_device("auto")
set_seed(seed)

sensor_types = ["Segment Velocity", "Segment Acceleration"]
video_suffix = None
window_size = 60
stride = 30
batch_size = 32
num_workers = 0
val_split = 0.2

---
print("[phase=data] loading P1 + P2 datasets")
DATA_DIR = PROJECT_ROOT / "output_data"

# P1 source
p1_dfs = [
    pd.read_csv(DATA_DIR / "P1_boning.csv"),
    pd.read_csv(DATA_DIR / "P1_slicing.csv"),
]
p1_raw_df = pd.concat(p1_dfs, ignore_index=True)

# P2 source
p2_dfs = [
    pd.read_csv(DATA_DIR / "P2_boning.csv"),
    pd.read_csv(DATA_DIR / "P2_slicing.csv"),
]
p2_raw_df = pd.concat(p2_dfs, ignore_index=True)

# Keep velocity + acceleration only
p1_raw_df = p1_raw_df[p1_raw_df["sensor_type"].isin(sensor_types)]
p2_raw_df = p2_raw_df[p2_raw_df["sensor_type"].isin(sensor_types)]

# Optional video suffix filter
if video_suffix is not None:
    p1_raw_df = p1_raw_df[p1_raw_df["video_id"].str.endswith(video_suffix)]
    p2_raw_df = p2_raw_df[p2_raw_df["video_id"].str.endswith(video_suffix)]

def merge_velocity_and_acceleration(df):
    vel_df = df[df["sensor_type"] == "Segment Velocity"].copy()
    acc_df = df[df["sensor_type"] == "Segment Acceleration"].copy()

    if vel_df.empty or acc_df.empty:
        raise ValueError("Both Segment Velocity and Segment Acceleration rows are required.")

    base_feature_cols = get_feature_columns(vel_df)

    id_cols = [
        c for c in ["video_id", "Frame", "Label", "person_id", "activity_type", "knife_sharpness_score"]
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

p1_df, base_feature_cols = merge_velocity_and_acceleration(p1_raw_df)
p2_df, _ = merge_velocity_and_acceleration(p2_raw_df)

velocity_feature_cols = [f"{c}_vel" for c in base_feature_cols]
acceleration_feature_cols = [f"{c}_acc" for c in base_feature_cols]
feature_cols = velocity_feature_cols + acceleration_feature_cols

assert p1_df["Label"].nunique() > 1, "Only one class left after filtering - train/val invalid"
assert p2_df["Label"].nunique() > 1, "Only one class left after filtering - test invalid"

print(f"[phase=data] p1_rows={len(p1_df)} p2_rows={len(p2_df)}")
print(f"[phase=data] velocity_features={len(velocity_feature_cols)} acceleration_features={len(acceleration_feature_cols)} total_features={len(feature_cols)}")
p1_df.head()

---
import matplotlib.pyplot as plt

# choose feature to visualize
feature_name = "L5 x_vel"   # change to any feature column

# optionally select one video from each dataset
p1_sample = p1_df[p1_df["video_id"] == p1_df["video_id"].iloc[0]]
p2_sample = p2_df[p2_df["video_id"] == p2_df["video_id"].iloc[0]]

# frame index
x1 = range(len(p1_sample))
x2 = range(len(p2_sample))

# create horizontal stacked plots
fig, axes = plt.subplots(2, 1, figsize=(12,6), sharex=False)

# P1 plot
axes[0].plot(x1, p1_sample[feature_name])
axes[0].set_title(f"P1 - {feature_name}")
axes[0].set_xlabel("Frame")
axes[0].set_ylabel("Feature value")

# P2 plot
axes[1].plot(x2, p2_sample[feature_name])
axes[1].set_title(f"P2 - {feature_name}")
axes[1].set_xlabel("Frame")
axes[1].set_ylabel("Feature value")

plt.tight_layout()
plt.show()
---
import pandas as pd

print("[phase=data] Unique labels and their counts per activity:")

# 1. Combine both datasets to get the overall count
combined_df = pd.concat([p1_df, p2_df], ignore_index=True)

# 2. Group by activity_type and Label to calculate counts
overall_label_counts = combined_df.groupby(
    ['Label']
).size().reset_index(name='Count')

# 3. Print the detailed breakdown
print(overall_label_counts.to_string(index=False))

---
print("[phase=data] Cleaning inconsistent labels")

label_mapping = {
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

p1_df['Label'] = p1_df['Label'].replace(label_mapping)
p2_df['Label'] = p2_df['Label'].replace(label_mapping)

for df in [p1_df, p2_df]:
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce')

for df in [p1_df, p2_df]:
    boning_mask = df['activity_type'].str.lower().eq('boning')
    df.loc[boning_mask & (df['Label'] == 5), 'Label'] = 8

for df in [p1_df, p2_df]:
    if df['Label'].isna().any():
        print("[warning] Found non-numeric labels after mapping")
    df['Label'] = df['Label'].astype('int64')

combined_df = pd.concat([p1_df, p2_df], ignore_index=True)

print("[phase=data] Unique video_id values after relabel:")
unique_video_ids = sorted(combined_df['video_id'].dropna().astype(str).unique().tolist())
print(f"total_video_ids={len(unique_video_ids)}")
for video_id in unique_video_ids:
    print(video_id)

overall_label_counts = combined_df.groupby(['activity_type', 'Label']).size().reset_index(name='Count')
print("\n[phase=data] label counts by activity:")
print(overall_label_counts.to_string(index=False))
---
import numpy as np
import pandas as pd

combined_df = pd.concat([p1_df, p2_df], ignore_index=True)

# Build segment list from base xyz feature names (e.g., "L5 x")
segment_names = []
for col in base_feature_cols:
    segment_name, axis = col.rsplit(" ", 1)
    if axis == "x":
        segment_names.append(segment_name)

# 22 velocity magnitude features + 22 acceleration magnitude features
for seg in segment_names:
    vx_col, vy_col, vz_col = f"{seg} x_vel", f"{seg} y_vel", f"{seg} z_vel"
    ax_col, ay_col, az_col = f"{seg} x_acc", f"{seg} y_acc", f"{seg} z_acc"

    vel_mag_col = f"{seg}_vel_mag"
    acc_mag_col = f"{seg}_acc_mag"

    combined_df[vel_mag_col] = np.sqrt(
        combined_df[vx_col] ** 2 + combined_df[vy_col] ** 2 + combined_df[vz_col] ** 2
    )
    combined_df[acc_mag_col] = np.sqrt(
        combined_df[ax_col] ** 2 + combined_df[ay_col] ** 2 + combined_df[az_col] ** 2
    )

velocity_mag_feature_cols = [f"{seg}_vel_mag" for seg in segment_names]
acceleration_mag_feature_cols = [f"{seg}_acc_mag" for seg in segment_names]

# 1 total body energy feature: sum of squared x,y,z over all joints (velocity + acceleration)
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
combined_df["total_body_energy"] = np.sum(np.column_stack(energy_terms), axis=1)

# Final training features: 22 + 22 + 1 = 45
feature_cols = velocity_mag_feature_cols + acceleration_mag_feature_cols + ["total_body_energy"]

print("[phase=features] Using engineered features by request")
print(f"Velocity magnitude features: {len(velocity_mag_feature_cols)}")
print(f"Acceleration magnitude features: {len(acceleration_mag_feature_cols)}")
print("Total body energy features: 1")
print(f"Total training features: {len(feature_cols)}")
print("Sample features:", feature_cols[:5], "...", feature_cols[-3:])
---
print(p1_df["Label"].dtype)
---
import pandas as pd

print("[phase=runs] Building contiguous label runs by video_id + label change (no groupby)")

combined_df = combined_df.sort_values(["video_id", "Frame"]).reset_index(drop=True)

runs = []
frame_counts = {}
run_counts = {}

start_idx = 0
current_video = combined_df.loc[0, "video_id"]
current_label = int(combined_df.loc[0, "Label"])

for i in range(len(combined_df)):
    label_i = int(combined_df.loc[i, "Label"])
    frame_counts[label_i] = frame_counts.get(label_i, 0) + 1

for i in range(1, len(combined_df)):
    video_i = combined_df.loc[i, "video_id"]
    label_i = int(combined_df.loc[i, "Label"])

    is_boundary = (video_i != current_video) or (label_i != current_label)

    if is_boundary:
        end_idx = i - 1
        start_frame = int(combined_df.loc[start_idx, "Frame"])
        end_frame = int(combined_df.loc[end_idx, "Frame"])
        run_length = end_idx - start_idx + 1

        runs.append({
            "video_id": current_video,
            "label": current_label,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "run_length": run_length,
        })

        run_counts[current_label] = run_counts.get(current_label, 0) + 1

        start_idx = i
        current_video = video_i
        current_label = label_i

end_idx = len(combined_df) - 1
start_frame = int(combined_df.loc[start_idx, "Frame"])
end_frame = int(combined_df.loc[end_idx, "Frame"])
run_length = end_idx - start_idx + 1
runs.append({
    "video_id": current_video,
    "label": current_label,
    "start_idx": start_idx,
    "end_idx": end_idx,
    "start_frame": start_frame,
    "end_frame": end_frame,
    "run_length": run_length,
})
run_counts[current_label] = run_counts.get(current_label, 0) + 1

print(f"[phase=runs] total_runs={len(runs)}")
print("[phase=runs] frame count per label:")
for label in sorted(frame_counts):
    print(f"  label {label}: {frame_counts[label]}")

print("[phase=runs] run count per label:")
for label in sorted(run_counts):
    print(f"  label {label}: {run_counts[label]}")

runs_df = pd.DataFrame(runs)
print("[phase=runs] run length stats:")
print(runs_df['run_length'].describe(percentiles=[0.5]).to_string())
---
import matplotlib.pyplot as plt

print("[phase=plot] Plotting one sample video with label transitions")

# Select first unique video from combined_df
sample_video_id = combined_df["video_id"].unique()[0]
sample_df = combined_df[combined_df["video_id"] == sample_video_id].copy()
sample_df = sample_df.sort_values("Frame").reset_index(drop=True)

print(f"[phase=plot] Sample video: {sample_video_id} ({len(sample_df)} frames)")

# Plot setup
feature_name = "L5 x_vel"  # Choose a sample feature to visualize
fig, ax = plt.subplots(figsize=(14, 5))

# Plot feature value vs Frame
frames = sample_df["Frame"].values
feature_values = sample_df[feature_name].values
ax.plot(frames, feature_values, linewidth=1.5, label=feature_name)

# Mark transition boundaries
transitions = sample_df[sample_df["Label"].diff().fillna(0) != 0]["Frame"].values
for t_frame in transitions:
    ax.axvline(x=t_frame, color='red', linestyle='--', alpha=0.6, linewidth=1.2)

# Shade regions by label
previous_transition = frames[0]
for t_frame in transitions:
    ax.axvspan(previous_transition, t_frame, alpha=0.1)
    previous_transition = t_frame
ax.axvspan(previous_transition, frames[-1], alpha=0.1)

ax.set_xlabel("Frame", fontsize=11)
ax.set_ylabel(f"{feature_name} (value)", fontsize=11)
ax.set_title(f"Sample Video: {sample_video_id} - Feature vs Frame with Label Transitions", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

print(f"[phase=plot] Red dashed lines indicate label transition points.")

---
# Normalize data (frame-level) using train frames only
from sklearn.preprocessing import StandardScaler

# Hold-out specific videos for validation (to avoid leakage)
TEST_VIDEO_IDS = ["MVN-J-Boning-90-004", "MVN-S-Slicing-63-001","MVN-S-Boning-89-004"]

# Split by video_id
train_df = combined_df[~combined_df["video_id"].isin(TEST_VIDEO_IDS)].copy()
val_df = combined_df[combined_df["video_id"].isin(TEST_VIDEO_IDS)].copy()

# Fit scaler on train frames only
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])

print("[phase=norm] StandardScaler fit on train frames only")
print("Train stats -> min:", train_df[feature_cols].min().min(),
      "max:", train_df[feature_cols].max().max(),
      "mean:", train_df[feature_cols].mean().mean(),
      "std:", train_df[feature_cols].std().mean())
# print("Val   stats -> min:", val_df[feature_cols].min().min(),
#       "max:", val_df[feature_cols].max().max(),
#       "mean:", val_df[feature_cols].mean().mean(),
#       "std:", val_df[feature_cols].std().mean())


print("[phase=norm] Train per-column min/max:")
train_min = train_df[feature_cols].min()
train_max = train_df[feature_cols].max()
print(pd.DataFrame({"min": train_min, "max": train_max}).to_string())

print("[phase=norm] Val per-column min/max:")
val_min = val_df[feature_cols].min()
val_max = val_df[feature_cols].max()

print(pd.DataFrame({"min": val_min, "max": val_max}).to_string(),
      "max:", val_df[feature_cols].max().max(),
      "mean:", val_df[feature_cols].mean().mean(),
      "std:", val_df[feature_cols].std().mean())



---
import numpy as np

print("[phase=windows] Building windows from contiguous runs with label-specific sampling")

# Default behavior for all labels.
DEFAULT_WINDOW_SIZE = 60
DEFAULT_WINDOW_STRIDE = 30
DEFAULT_FRAME_SKIP_STEP = 2  # keep every 2nd frame -> skip 1 frame between selected frames

# Label-specific overrides.
# - Label 4: window=60, stride=20, skip-frame=2 -> sampling step=3.
# - Labels 0, 6, 7, 8: window=60, stride=30, no skip -> sampling step=1.
LABEL_SPECIFIC_RULES = {
    4: {
        "window_size": 60,
        "window_stride": 30,
        "frame_skip_step": 3,
    },
    0: {
        "window_size": 60,
        "window_stride": 30,
        "frame_skip_step": 2,
    },
    6: {
        "window_size": 60,
        "window_stride": 15,
        "frame_skip_step": 1,
    },
    7: {
        "window_size": 60,
        "window_stride": 20,
        "frame_skip_step": 1,
    },
    5: {
        "window_size": 60,
        "window_stride": 20,
        "frame_skip_step": 1,
    },
    8: {
        "window_size": 60,
        "window_stride": 20,
        "frame_skip_step": 2,
    },
}

if not hasattr(scaler, "mean_"):
    raise RuntimeError("Scaler is not fitted. Run the Normalize Data cell first.")

combined_df_norm = combined_df.copy()
combined_df_norm[feature_cols] = scaler.transform(combined_df[feature_cols])

if not np.isfinite(combined_df_norm[feature_cols].to_numpy()).all():
    raise ValueError("Normalization produced non-finite values. Check feature columns and scaler fit.")

X_windows = []
y_windows = []
window_meta = []
window_lengths = []

for run_idx, run in enumerate(runs):
    label = int(run["label"])

    params = LABEL_SPECIFIC_RULES.get(
        label,
        {
            "window_size": DEFAULT_WINDOW_SIZE,
            "window_stride": DEFAULT_WINDOW_STRIDE,
            "frame_skip_step": DEFAULT_FRAME_SKIP_STEP,
        },
    )

    window_size = int(params["window_size"])
    window_stride = int(params["window_stride"])
    frame_skip_step = int(params["frame_skip_step"])

    run_indices = np.arange(run["start_idx"], run["end_idx"] + 1)
    sampled_indices = run_indices[::frame_skip_step]

    if len(sampled_indices) == 0:
        continue

    if len(sampled_indices) <= window_size:
        idx_chunk = sampled_indices
        X_windows.append(combined_df_norm.iloc[idx_chunk][feature_cols].to_numpy())
        y_windows.append(label)
        window_lengths.append(len(idx_chunk))
        window_meta.append({
            "run_idx": run_idx,
            "video_id": run["video_id"],
            "label": label,
            "start_frame": int(combined_df.iloc[idx_chunk[0]]["Frame"]),
            "end_frame": int(combined_df.iloc[idx_chunk[-1]]["Frame"]),
            "window_len": len(idx_chunk),
            "window_size": window_size,
            "window_stride": window_stride,
            "frame_skip_step": frame_skip_step,
        })
        continue

    starts = list(range(0, len(sampled_indices) - window_size + 1, window_stride))

    for start in starts:
        idx_chunk = sampled_indices[start:start + window_size]
        X_windows.append(combined_df_norm.iloc[idx_chunk][feature_cols].to_numpy())
        y_windows.append(label)
        window_lengths.append(len(idx_chunk))
        window_meta.append({
            "run_idx": run_idx,
            "video_id": run["video_id"],
            "label": label,
            "start_frame": int(combined_df.iloc[idx_chunk[0]]["Frame"]),
            "end_frame": int(combined_df.iloc[idx_chunk[-1]]["Frame"]),
            "window_len": len(idx_chunk),
            "window_size": window_size,
            "window_stride": window_stride,
            "frame_skip_step": frame_skip_step,
        })

    last_covered = starts[-1] + window_size
    if last_covered < len(sampled_indices):
        idx_chunk = sampled_indices[last_covered:]
        X_windows.append(combined_df_norm.iloc[idx_chunk][feature_cols].to_numpy())
        y_windows.append(label)
        window_lengths.append(len(idx_chunk))
        window_meta.append({
            "run_idx": run_idx,
            "video_id": run["video_id"],
            "label": label,
            "start_frame": int(combined_df.iloc[idx_chunk[0]]["Frame"]),
            "end_frame": int(combined_df.iloc[idx_chunk[-1]]["Frame"]),
            "window_len": len(idx_chunk),
            "window_size": window_size,
            "window_stride": window_stride,
            "frame_skip_step": frame_skip_step,
        })

y_windows = np.array(y_windows, dtype=np.int64)
window_meta_df = pd.DataFrame(window_meta)

window_count_per_label = {}
for label in y_windows.tolist():
    window_count_per_label[label] = window_count_per_label.get(label, 0) + 1

print(f"[phase=windows] total_windows={len(X_windows)}")
print("[phase=windows] windows per label:")
for label in sorted(window_count_per_label):
    print(f"  label {label}: {window_count_per_label[label]}")

window_lengths_np = np.array(window_lengths)
short_count = int((window_lengths_np < DEFAULT_WINDOW_SIZE).sum())
print(f"[phase=windows] short_windows(<{DEFAULT_WINDOW_SIZE})={short_count}")
print(
    f"[phase=windows] window length min/median/max = "
    f"{window_lengths_np.min()}/{int(np.median(window_lengths_np))}/{window_lengths_np.max()}"
)

if not window_meta_df.empty:
    print("[phase=windows] Effective sampling settings by label:")
    settings_by_label = (
        window_meta_df[["label", "window_size", "window_stride", "frame_skip_step"]]
        .drop_duplicates()
        .sort_values("label")
    )
    print(settings_by_label.to_string(index=False))
---
import numpy as np

print("[phase=split] Split windows by TEST_VIDEO_IDS and pad to length 60")

train_mask = ~window_meta_df["video_id"].isin(TEST_VIDEO_IDS)
val_mask = window_meta_df["video_id"].isin(TEST_VIDEO_IDS)

train_indices = np.where(train_mask.to_numpy())[0]
val_indices = np.where(val_mask.to_numpy())[0]

X_train_list = [X_windows[i] for i in train_indices]
y_train_all = y_windows[train_indices]
X_val_list = [X_windows[i] for i in val_indices]
y_val = y_windows[val_indices]

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

X_train_all = pad_windows_to_60(X_train_list, target_len=60)
X_val = pad_windows_to_60(X_val_list, target_len=60)

print(f"[phase=split] Train windows: {X_train_all.shape}, y_train: {y_train_all.shape}")
print(f"[phase=split] Val windows:   {X_val.shape}, y_val:   {y_val.shape}")
---
# Use full train windows (no window-level split)
X_train, y_train = X_train_all, y_train_all

print("Train X:", X_train.shape, "Train y:", y_train.shape)
print("Val   X:", X_val.shape, "Val   y:", y_val.shape)


---
# Focal Loss for class imbalance (gamma=2.0)
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

criterion = FocalLoss(gamma=2.0)
print("[phase=loss] Using FocalLoss gamma=2.0")


---
# Encode labels to 0..C-1
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_all)
y_val = label_encoder.transform(y_val)

print("Encoded classes:", label_encoder.classes_)
print("y_train unique (encoded):", np.unique(y_train))
print("y_val unique (encoded):", np.unique(y_val))
---
import numpy as np

print("[phase=dataset] Augmentation disabled")

X_train_aug = X_train_all
y_train_aug = y_train

print(f"[phase=dataset] X_train_aug shape: {X_train_aug.shape}")
print(f"[phase=dataset] y_train_aug shape: {y_train_aug.shape}")
---
import torch
import torch.nn as nn
from tqdm import tqdm # Optional: for progress bars during epochs

print("[phase=model] Defining BiLSTM Architecture")

# 1. Define the BiLSTM Model Class
class ActivityBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.4):
        super(ActivityBiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Single FC block: BN -> Linear -> Dropout -> ReLU
        self.fc_bn = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Final output layer
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length/window_size, input_size/features)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Temporal pooling over time dimension (mean pooling)
        # out shape: (batch_size, seq_len, hidden_size * 2)
        out = out.mean(dim=1)
        # out = out[:, -1, :]

        # FC block
        out = self.fc_bn(out)
        out = self.fc(out)
        out = self.dropout(out)
        out = self.relu(out)

        # Final output layer
        out = self.fc_out(out)
        return out


---
# Instantiate model, criterion, optimizer
input_size = X_train_aug.shape[-1]
num_classes = len(label_encoder.classes_)

hidden_size = 128
num_layers = 1
dropout = 0.4

model = ActivityBiLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout=dropout,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

print(f"[phase=model] Model params: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, num_classes={num_classes}")




---
# Train loop with early stopping, checkpoint, and ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

# Convert to torch tensors
X_train_t = torch.tensor(X_train_aug, dtype=torch.float32)
y_train_t = torch.tensor(y_train_aug, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

# Reuse validation set for final evaluation
X_test_t = X_val_t
y_test_t = y_val_t

# # Weighted sampling to balance classes during training
# train_class_counts = np.bincount(y_train)
# train_class_weights = 1.0 / np.maximum(train_class_counts, 1)
# sample_weights = train_class_weights[y_train]
# sampler = torch.utils.data.WeightedRandomSampler(
#     weights=torch.tensor(sample_weights, dtype=torch.double),
#     num_samples=len(sample_weights),
#     replacement=True,
# )

train_batch_size = 32
if len(X_train_t) < 2:
    raise ValueError(
        "Training set has fewer than 2 samples. BatchNorm1d requires at least 2 samples per training batch."
    )

# Keep BatchNorm stable by preventing a final batch of size 1.
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=train_batch_size,
    shuffle=True,
    drop_last=True,
)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=train_batch_size, shuffle=False)

if len(train_loader) == 0:
    raise ValueError(
        "No training batches were created. Reduce train_batch_size or increase training data."
    )

print(
    f"[phase=train] train_samples={len(X_train_t)} batch_size={train_batch_size} "
    f"drop_last=True train_batches={len(train_loader)}"
)
if len(X_train_t) % train_batch_size == 1:
    print("[phase=train] Dropping final single-sample batch to avoid BatchNorm1d training error.")

# Callbacks (manual implementations)
early_stopping_patience = 10
best_val_loss = float('inf')
patience_counter = 0

checkpoint_path = PROJECT_ROOT  / "notebook" / "best_bilstm.pt"

# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, T_0=10, T_mult=2
# )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

prev_lr = optimizer.param_groups[0]['lr']

def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

num_epochs = 30

for epoch in range(1, num_epochs + 1):
    # Train
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    processed_train_samples = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size_now = xb.size(0)
        train_loss += loss.item() * batch_size_now
        train_acc += accuracy_from_logits(logits, yb) * batch_size_now
        processed_train_samples += batch_size_now

    train_loss /= processed_train_samples
    train_acc /= processed_train_samples

    # Validate
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss += loss.item() * xb.size(0)
            val_acc += accuracy_from_logits(logits, yb) * xb.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)

    # Reduce LR on plateau
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != prev_lr:
        print(f"[lr] {prev_lr:.6f} -> {new_lr:.6f}")
        prev_lr = new_lr

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Early stopping + checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), checkpoint_path)
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
---
# Test evaluation (held-out videos)
print("[phase=eval] Evaluating on validation videos")

# Load best checkpoint if available
if checkpoint_path.exists():
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

model.eval()

test_loss = 0.0
test_acc = 0.0

test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        test_loss += loss.item() * xb.size(0)
        test_acc += accuracy_from_logits(logits, yb) * xb.size(0)

if len(test_loader.dataset) > 0:
    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
else:
    print("[warning] No test samples after filtering unknown labels.")


---
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

print("[phase=eval] Classification report + confusion matrix")

# Reload best checkpoint for consistent evaluation.
if checkpoint_path.exists():
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

model.eval()
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(yb.cpu().numpy().tolist())

if len(all_targets) == 0:
    print("[warning] No test samples available for classification report/confusion matrix.")
else:
    labels = np.arange(len(label_encoder.classes_))
    class_names = [str(c) for c in label_encoder.classes_]

    print("[phase=eval] Classification report:")
    print(
        classification_report(
            all_targets,
            all_preds,
            labels=labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    )

    print("[phase=eval] Confusion matrix (counts):")
    print(cm_df.to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
---
import pandas as pd
import numpy as np

# Label distribution: train vs val windows
train_labels, train_counts = np.unique(y_train, return_counts=True)
val_labels, val_counts = np.unique(y_val, return_counts=True)

train_dist = dict(zip(label_encoder.inverse_transform(train_labels), train_counts))
val_dist = dict(zip(label_encoder.inverse_transform(val_labels), val_counts))

all_classes = sorted(set(train_dist) | set(val_dist))
print(f"{'Class':<6} {'Train':>8} {'Val':>8} {'Val%':>8}")
print("-" * 34)
for c in all_classes:
    t = train_dist.get(c, 0)
    v = val_dist.get(c, 0)
    pct = v / sum(val_counts) * 100
    print(f"{c:<6} {t:>8} {v:>8} {pct:>7.1f}%")
---

---
