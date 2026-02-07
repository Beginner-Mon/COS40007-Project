import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
import pickle

from hydra.utils import get_original_cwd

from utils.seed import set_seed
from utils.device import get_device

from data.preprocessing import (
    get_feature_columns,
    create_windows,
    clean_features,
)
from data.motion_dataset import MotionDataset
from model.bilstm import BiLSTM
from training.loss import build_loss


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):

    # ======================================================
    # Setup
    # ======================================================
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    PROJECT_ROOT = Path(get_original_cwd())
    DATA_DIR = PROJECT_ROOT / "output_data"

    # ======================================================
    # Load model + artifacts (from training run or shared dir)
    # ======================================================
    model_path = Path(cfg.model_path)
    run_dir = model_path.parent

    artifacts_dir = None
    if cfg.artifacts_dir:
        artifacts_dir = Path(cfg.artifacts_dir)
        if not artifacts_dir.is_absolute():
            artifacts_dir = PROJECT_ROOT / artifacts_dir

    source_dir = artifacts_dir if artifacts_dir else run_dir
    print(f"[artifacts] loading from: {source_dir}")

    with open(source_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(source_dir / "label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    checkpoint = torch.load(model_path, map_location=device)

    # ======================================================
    # Load evaluation data (participant selection)
    # ======================================================
    participant = str(cfg.data.participant).lower()
    if participant not in ("p1", "p2", "both"):
        raise ValueError(f"Invalid participant: {cfg.data.participant}")
    participant_ids = ["P1", "P2"] if participant == "both" else [participant.upper()]

    dfs = []
    for pid in participant_ids:
        dfs.append(pd.read_csv(DATA_DIR / f"{pid}_boning.csv"))
        dfs.append(pd.read_csv(DATA_DIR / f"{pid}_slicing.csv"))

    df = pd.concat(dfs, ignore_index=True)

    # filter sensor
    df = df[df["sensor_type"] == cfg.data.sensor_type]

    # optional video suffix
    if cfg.data.video_suffix is not None:
        df = df[df["video_id"].str.endswith(cfg.data.video_suffix)]

    assert df["activity_type"].nunique() > 1, \
        "Only one class left after filtering — evaluation invalid"

    # ======================================================
    # Preprocessing (IDENTICAL to training)
    # ======================================================
    feature_cols = get_feature_columns(df)

    X, y = create_windows(df, feature_cols, cfg.data.window_size, cfg.data.stride)
    X = clean_features(X)

    # normalize using training scaler
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    y = encoder.transform(y)

    # ======================================================
    # Dataset / Loader
    # ======================================================
    dataset = MotionDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    # ======================================================
    # Model
    # ======================================================
    model = BiLSTM(
        input_size=X.shape[2],
        hidden_size=cfg.model.hidden_size,
        num_classes=len(encoder.classes_),
        num_layers=cfg.model.num_layers
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    criterion, mode = build_loss(cfg.loss, len(encoder.classes_))

    # ======================================================
    # Evaluation loop
    # ======================================================
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            if mode == "ce":
                loss = criterion(outputs, y_batch)
                preds = outputs.argmax(dim=1)
            else:
                loss = criterion(outputs.squeeze(), y_batch.float())
                preds = (outputs > 0.5).long()

            total_loss += loss.item() * y_batch.size(0)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    eval_subject = "+".join(participant_ids)
    print(f"[eval] subject={eval_subject}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")

    # ======================================================
    # Save results
    # ======================================================
    results = {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "num_samples": int(total),
        "checkpoint": model_path.name,
        "eval_subject": eval_subject,
    }

    with open(run_dir / f"eval_results_{eval_subject}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(results))

    print(f"Evaluation complete. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
