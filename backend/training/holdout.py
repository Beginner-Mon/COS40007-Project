import torch
from omegaconf import OmegaConf
import mlflow
from sklearn.model_selection import GroupShuffleSplit

from training.core_run import run_training_core


def run_holdout_training(X_all, y_all, window_meta_df, cfg, run_dir, label_encoder, device):
    print("[START] Launching Holdout (Train/Test Split) Validation...", flush=True)
    test_vids = list(cfg.task.get("test_video_ids", []))
    holdout_mode = "manual"

    if test_vids:
        train_mask = ~window_meta_df["video_id"].isin(test_vids)
        val_mask = window_meta_df["video_id"].isin(test_vids)

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_va, y_va = X_all[val_mask], y_all[val_mask]

        if len(X_va) == 0:
            raise ValueError(
                f"Holdout validation set is empty. Check if TEST_VIDEO_IDS {test_vids} exist in the loaded data."
            )
    else:
        holdout_mode = "auto_group_split"
        groups = window_meta_df["video_id"].astype(str).to_numpy()
        unique_groups = sorted(window_meta_df["video_id"].astype(str).unique().tolist())
        if len(unique_groups) < 2:
            raise ValueError(
                "Holdout requires at least 2 unique video_id groups for train/validation split."
            )

        test_size = float(cfg.data.get("val_split", 0.2))
        if test_size <= 0 or test_size >= 1:
            test_size = 0.2

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=cfg.seed)
        train_idx, val_idx = next(splitter.split(X_all, y_all, groups=groups))

        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_va, y_va = X_all[val_idx], y_all[val_idx]

        selected_val_vids = sorted(set(groups[val_idx].tolist()))
        preview = ", ".join(selected_val_vids[:5])
        suffix = "..." if len(selected_val_vids) > 5 else ""
        print(
            "[INFO] 'task.test_video_ids' not set. "
            f"Using automatic group holdout (val_split={test_size}) with {len(selected_val_vids)} validation videos: "
            f"{preview}{suffix}",
            flush=True,
        )

    if len(X_tr) == 0:
        raise ValueError("Holdout training set is empty after split. Adjust test_video_ids or val_split.")
    if len(X_va) == 0:
        raise ValueError("Holdout validation set is empty after split. Adjust test_video_ids or val_split.")

    val_loss, model, scaler = run_training_core(
        X_tr, y_tr, X_va, y_va, cfg, label_encoder, run_dir, device, fold=1
    )
    
    if mlflow.active_run():
        mlflow.log_metric("val_loss", float(val_loss))
        mlflow.log_param("holdout_mode", holdout_mode)
    print(f"[DONE] Holdout finished. Best Val Loss: {val_loss:.4f}", flush=True)

    save_path = run_dir / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_encoder": label_encoder,
        "scaler": scaler,
        "config": OmegaConf.to_container(cfg)
    }, save_path)
    print(f"Saved best Holdout model to {save_path.resolve()}", flush=True)
