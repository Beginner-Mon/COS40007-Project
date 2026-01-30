import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from pathlib import Path

from model.bilstm import BiLSTM
from data.datasets import MotionDataset

# ======================
# CONFIG
# ======================
WINDOW_SIZE = 30
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
HIDDEN_SIZE = 128
GRAD_CLIP = 1.0

EARLY_STOP_PATIENCE = 8
LR_PATIENCE = 4
LR_FACTOR = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

NON_FEATURE_COLS = [
    "Frame", "Label", "person_id", "activity_type",
    "knife_sharpness_score", "sensor_type", "video_id"
]

# ======================
# DATA LOADING
# ======================
def load_data():
    df_boning = pd.read_csv("output_data/P1_boning.csv")
    df_slicing = pd.read_csv("output_data/P1_slicing.csv")

    df = pd.concat([df_boning, df_slicing], ignore_index=True)
    df = df[df["sensor_type"] == "Segment Velocity"]

    print("Number of videos:", df["video_id"].nunique())
    return df


def get_feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# ======================
# WINDOWING
# ======================
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


# ======================
# PREPROCESSING
# ======================
def preprocess(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    B, T, F = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, F)).reshape(B, T, F)

    return X, scaler


# ======================
# METRICS
# ======================
def accuracy_from_logits(logits, y, is_binary):
    if is_binary:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long().squeeze(1)
    else:
        preds = torch.argmax(logits, dim=1)

    return (preds == y).float().mean().item()


# ======================
# TRAIN / VALIDATE
# ======================
def run_epoch(model, loader, optimizer, criterion, is_binary, train=True):
    model.train() if train else model.eval()

    total_loss, total_acc = 0.0, 0.0

    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            if is_binary:
                y = y.float().unsqueeze(1)

            logits = model(X)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy_from_logits(
                logits, y.long().squeeze(), is_binary
            )

    return total_loss / len(loader), total_acc / len(loader)


# ======================
# MAIN
# ======================
def main():
    df = load_data()
    feature_cols = get_feature_columns(df)

    X, y = create_windows(df, feature_cols, WINDOW_SIZE)
    X, scaler = preprocess(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    num_classes = len(le.classes_)
    is_binary = num_classes == 2

    print("Classes:", le.classes_)
    print("Distribution:", Counter(y))

    # Train / validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_loader = DataLoader(
        MotionDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        MotionDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = BiLSTM(
        input_size=X.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_classes=num_classes
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    criterion = (
        nn.BCEWithLogitsLoss() if is_binary
        else nn.CrossEntropyLoss()
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE
    )

    writer = SummaryWriter("runs/bilstm")

    best_val_acc = 0.0
    patience_counter = 0

    print("Training started...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion, is_binary, train=True
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, criterion, is_binary, train=False
        )

        scheduler.step(val_acc)

        writer.add_scalars(
            "Loss", {"train": train_loss, "val": val_loss}, epoch
        )
        writer.add_scalars(
            "Accuracy", {"train": train_acc, "val": val_acc}, epoch
        )

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

        # 🔥 Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler": scaler,
                "label_encoder": le
            }, CKPT_DIR / "best_model.pth")

            print("✅ Best model saved")

        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("⏹ Early stopping triggered")
                break

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
