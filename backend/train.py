import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import traceback
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig

from utils.seed import set_seed
from utils.device import get_device
from utils.arg_parser import parse_runtime_args

from data.preprocessing import (
    get_feature_columns,
    create_windows,
    clean_features,
    normalize_features,
)
from data.motion_dataset import MotionDataset
from model.bilstm import BiLSTM
from training.loss import build_loss
from training.trainer import train_epoch
from callbacks.early_stopping import EarlyStopping
from callbacks.checkpoint import ModelCheckpoint
from callbacks.lr_scheduler import ReduceLROnPlateau


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):

    # ======================================================
    # Runtime args (argparse)
    # ======================================================
    runtime_args, _ = parse_runtime_args()

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
    df_boning = pd.read_csv(DATA_DIR / "P1_boning.csv")
    df_slicing = pd.read_csv(DATA_DIR / "P1_slicing.csv")

    df = pd.concat([df_boning, df_slicing], ignore_index=True)

    # filter sensor
    df = df[df["sensor_type"] == cfg.data.sensor_type]

    # optional: filter video suffix (e.g. 001)
    if cfg.data.video_suffix is not None:
        df = df[df["video_id"].str.endswith(cfg.data.video_suffix)]

    assert df["activity_type"].nunique() > 1, \
        "❌ Only one class left after filtering — task is invalid"

    # ======================================================
    # Windowing + preprocessing
    # ======================================================
    feature_cols = get_feature_columns(df)

    X, y = create_windows(df, feature_cols, cfg.data.window_size, cfg.data.stride)
    X = clean_features(X)
    X, scaler = normalize_features(X)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Save scaler and encoder for evaluation/inference
    pickle.dump(scaler, open(run_dir / "scaler.pkl", "wb"))
    pickle.dump(encoder, open(run_dir / "label_encoder.pkl", "wb"))

    # ======================================================
    # Dataset / Loader
    # ======================================================
    dataset = MotionDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    # ======================================================
    # Model / Optim / Loss
    # ======================================================
    model = BiLSTM(
        input_size=X.shape[2],
        hidden_size=cfg.model.hidden_size,
        num_classes=len(encoder.classes_),
        num_layers=cfg.model.num_layers
    ).to(device)

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

        loss, acc = train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=cfg.train.grad_clip,
            mode=mode
        )

        print(f"[{epoch+1}/{cfg.train.epochs}] "
              f"loss={loss:.4f} | acc={acc:.4f}")

        logs = {
            "loss": loss,
            "acc": acc,
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
