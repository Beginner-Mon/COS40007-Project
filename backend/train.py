import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import traceback
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig

from utils.seed import set_seed
from utils.device import get_device

from data.preprocessing import (
    get_feature_columns,
    create_windows,
    clean_features,
    normalize_features,
)
from data.motion_dataset import MotionDataset
from model.bilstm import BiLSTM
from training.loss import build_loss
from training.trainer import train_epoch, eval_epoch
from callbacks.early_stopping import EarlyStopping
from callbacks.checkpoint import ModelCheckpoint
from callbacks.lr_scheduler import ReduceLROnPlateau


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):

    run_dir = Path(HydraConfig.get().runtime.output_dir)

    # save config actually used
    with open(run_dir / "config_used.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # ======================================================
    # Setup
    # ======================================================
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    PROJECT_ROOT = Path(get_original_cwd())
    DATA_DIR = PROJECT_ROOT / "output_data"

    # ======================================================
    # Load data (BONING + SLICING)
    # ======================================================
    participant = str(cfg.data.participant).lower()
    if participant not in ("p1", "p2", "both"):
        raise ValueError(f"Invalid participant: {cfg.data.participant}")
    participant_ids = ["P1", "P2"] if participant == "both" else [participant.upper()]

    print(f"[phase=data] participant={participant} | sensor={cfg.data.sensor_type} | "
          f"window={cfg.data.window_size} | stride={cfg.data.stride}")

    dfs = []
    for pid in participant_ids:
        dfs.append(pd.read_csv(DATA_DIR / f"{pid}_boning.csv"))
        dfs.append(pd.read_csv(DATA_DIR / f"{pid}_slicing.csv"))

    df = pd.concat(dfs, ignore_index=True)
    print(f"[phase=data] loaded rows={len(df)} from {', '.join(participant_ids)}")

    # filter sensor
    df = df[df["sensor_type"] == cfg.data.sensor_type]

    # optional: filter video suffix (e.g. 001)
    if cfg.data.video_suffix is not None:
        df = df[df["video_id"].str.endswith(cfg.data.video_suffix)]

    assert df["activity_type"].nunique() > 1, \
        "❌ Only one class left after filtering — task is invalid"

    # ======================================================
    # Split by video_id FIRST (CRITICAL FIX)
    # ======================================================
    feature_cols = get_feature_columns(df)

    video_ids = df["video_id"].unique()

    train_vids, val_vids = train_test_split(
        video_ids,
        test_size=cfg.data.val_split,
        random_state=cfg.seed,
    )

    train_df = df[df["video_id"].isin(train_vids)].copy()
    val_df   = df[df["video_id"].isin(val_vids)].copy()

    print(f"[phase=split] train_videos={len(train_vids)} | val_videos={len(val_vids)}")

    # ======================================================
    # Create windows separately
    # ======================================================
    X_train, y_train = create_windows(
        train_df,
        feature_cols,
        cfg.data.window_size,
        cfg.data.stride
    )

    X_val, y_val = create_windows(
        val_df,
        feature_cols,
        cfg.data.window_size,
        cfg.data.stride
    )

    # ======================================================
    # Clean features
    # ======================================================
    X_train = clean_features(X_train)
    X_val   = clean_features(X_val)

    # ======================================================
    # Fit scaler ONLY on training data
    # ======================================================
    X_train, scaler = normalize_features(X_train)

    X_val = scaler.transform(
        X_val.reshape(-1, X_val.shape[-1])
    ).reshape(X_val.shape)

    print(f"[phase=preprocess] train_windows={len(X_train)} | val_windows={len(X_val)}")

    # ======================================================
    # Encode labels
    # ======================================================
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)
    y_val   = encoder.transform(y_val)


    train_dataset = MotionDataset(X_train, y_train)
    val_dataset = MotionDataset(X_val, y_val)
    print(f"[phase=split] train={len(train_dataset)} | val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    # ======================================================
    # Model / Optim / Loss
    # ======================================================
    model = BiLSTM(
    input_size=X_train.shape[2],
    hidden_size=cfg.model.hidden_size,
    num_classes=len(encoder.classes_),
    num_layers=cfg.model.num_layers
    ).to(device)

    print(f"[phase=model] type=BiLSTM | input={X_train.shape[2]} | hidden={cfg.model.hidden_size} | "
          f"layers={cfg.model.num_layers} | classes={len(encoder.classes_)} | device={device}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )

    criterion, mode = build_loss(cfg.loss, len(encoder.classes_))

    # ======================================================
    # Callbacks
    # ======================================================
    es = None
    if cfg.callbacks.early_stopping.enabled:
        es = EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
            min_delta=cfg.callbacks.early_stopping.min_delta
        )

    ckpt = None
    if cfg.callbacks.checkpoint.enabled:
        ckpt = ModelCheckpoint(
            monitor=cfg.callbacks.checkpoint.monitor,
            save_dir=run_dir,
            mode="max"
        )


    rlr = None
    if cfg.callbacks.reduce_lr.enabled:
        rlr = ReduceLROnPlateau(
            optimizer=optimizer,
            monitor=cfg.callbacks.reduce_lr.monitor,
            factor=cfg.callbacks.reduce_lr.factor,
            patience=cfg.callbacks.reduce_lr.patience,
            min_lr=cfg.callbacks.reduce_lr.min_lr
        )


    # ======================================================
    # Training loop
    # ======================================================
    for epoch in range(cfg.train.epochs):

        train_loss, train_acc = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=cfg.train.grad_clip,
            mode=mode
        )

        val_loss, val_acc = eval_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            mode=mode
        )

        print(f"[{epoch+1}/{cfg.train.epochs}] "
              f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        logs = {
            "loss": train_loss,
            "acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "model": model
        }

        if ckpt:
            ckpt.on_epoch_end(epoch, logs)

        if es:
            es.on_epoch_end(epoch, logs)
            if es.stop:
                print("🛑 Early stopping")
                break

        if rlr:
            rlr.on_epoch_end(epoch, logs)


    # ======================================================
    # Save final artifacts
    # ======================================================
    try:
        save_path = run_dir / "last_model.pt"
        print(f"Attempting to save last_model to: {save_path.resolve()}")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_encoder": encoder,
                "scaler": scaler,
                "config": OmegaConf.to_container(cfg)
            },
            save_path
        )
        print(f"Successfully saved last_model to: {save_path.resolve()}")
    except Exception as e:
        print(f"Error saving last_model: {e}")
        traceback.print_exc()

    print(f"✅ Training finished. Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
