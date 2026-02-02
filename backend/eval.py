import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle

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


@hydra.main(config_path="configs", config_name="eval", version_base=None)
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
    # Load saved artifacts
    # ======================================================
    model_path = Path(cfg.model_path)
    checkpoint = torch.load(model_path)
    model_dir = model_path.parent

    if 'model_state_dict' in checkpoint:
        # Loading from last_model.pt (full checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint['scaler']
        encoder = checkpoint['label_encoder']
        saved_cfg = OmegaConf.create(checkpoint['config'])
    else:
        # Loading from best.pt (state_dict only)
        # Load supporting artifacts from the run directory
        scaler = pickle.load(open(model_dir / "scaler.pkl", "rb"))
        encoder = pickle.load(open(model_dir / "label_encoder.pkl", "rb"))
        saved_cfg = OmegaConf.load(model_dir / "config_used.yaml")

    # ======================================================
    # Load test data (P2 BONING + SLICING)
    # ======================================================
    df_boning = pd.read_csv(DATA_DIR / "P2_boning.csv")
    df_slicing = pd.read_csv(DATA_DIR / "P2_slicing.csv")

    df = pd.concat([df_boning, df_slicing], ignore_index=True)

    # filter sensor (use from saved cfg to match training)
    df = df[df["sensor_type"] == saved_cfg.data.sensor_type]

    # optional: filter video suffix (use from saved cfg)
    if saved_cfg.data.video_suffix is not None:
        df = df[df["video_id"].str.endswith(saved_cfg.data.video_suffix)]

    assert df["activity_type"].nunique() > 1, \
        "❌ Only one class left after filtering — task is invalid"

    # ======================================================
    # Windowing + preprocessing (use saved scaler)
    # ======================================================
    feature_cols = get_feature_columns(df)

    X, y = create_windows(df, feature_cols, saved_cfg.data.window_size)
    X = clean_features(X)
    X = scaler.transform(X)  # Use saved scaler for normalization

    y = encoder.transform(y)  # Use saved encoder

    # ======================================================
    # Dataset / Loader
    # ======================================================
    dataset = MotionDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=saved_cfg.data.batch_size,
        shuffle=False,
        num_workers=saved_cfg.data.num_workers,
        pin_memory=True
    )

    # ======================================================
    # Model / Loss
    # ======================================================
    model = BiLSTM(
        input_size=X.shape[2],
        hidden_size=saved_cfg.model.hidden_size,
        num_classes=len(encoder.classes_),
        num_layers=saved_cfg.model.num_layers
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    criterion, mode = build_loss(saved_cfg.loss, len(encoder.classes_))

    # ======================================================
    # Evaluation loop
    # ======================================================
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            if mode == "multi":
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
            else:
                predicted = (outputs > 0.5).float().squeeze()
                total_correct += (predicted == labels).sum().item()

            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(f"Evaluation results: loss={avg_loss:.4f} | acc={accuracy:.4f}")

    # Optionally save results
    results = {"loss": avg_loss, "acc": accuracy}
    with open(run_dir / "eval_results.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(results))

    print(f"✅ Evaluation finished. Results saved to {run_dir}")


if __name__ == "__main__":
    main()