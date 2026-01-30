import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torch
from utils.seed import set_seed
from utils.device import get_device
from data.preprocessing import *
from data.motion_dataset import MotionDataset
from training.loss import build_loss
from training.trainer import train_epoch
from callbacks.early_stopping import EarlyStopping
from callbacks.checkpoint import ModelCheckpoint
from callbacks.lr_scheduler import ReduceLROnPlateau
from model.bilstm import BiLSTM


@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    df = pd.read_csv("output_data/P1_boning.csv")  # demo
    feature_cols = get_feature_columns(df)

    X, y = create_windows(df, feature_cols, cfg.data.window_size)
    X = clean_features(X)
    X, scaler = normalize_features(X)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    dataset = MotionDataset(X, y)
    loader = DataLoader(dataset, cfg.data.batch_size, shuffle=True)

    model = BiLSTM(
        input_size=X.shape[2],
        hidden_size=cfg.model.hidden_size,
        num_classes=len(encoder.classes_)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion, mode = build_loss(cfg.loss, len(encoder.classes_))

    es = EarlyStopping(**cfg.callbacks.early_stopping) \
        if cfg.callbacks.early_stopping.enabled else None

    ckpt = ModelCheckpoint("checkpoints/best.pt") \
        if cfg.callbacks.checkpoint.enabled else None

    rlr = ReduceLROnPlateau(optimizer, **cfg.callbacks.reduce_lr) \
        if cfg.callbacks.reduce_lr.enabled else None

    for epoch in range(cfg.train.epochs):
        loss, acc = train_epoch(
            model, loader, optimizer,
            criterion, device,
            cfg.train.grad_clip, mode
        )

        print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}")

        if ckpt:
            ckpt.step(acc, model)
        if es:
            es.step(acc)
            if es.stop:
                break
        if rlr:
            rlr.step(loss)


if __name__ == "__main__":
    main()
