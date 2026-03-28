import torch
from omegaconf import OmegaConf
import mlflow

from training.core_run import run_training_core


def run_holdout_training(X_all, y_all, window_meta_df, cfg, run_dir, label_encoder, device):
    print("🚀 Launching Holdout (Train/Test Split) Validation...")
    test_vids = list(cfg.task.get("test_video_ids", []))
    if not test_vids:
        raise ValueError("Holdout strategy selected but 'test_video_ids' is missing or empty in configuration.")
    
    train_mask = ~window_meta_df["video_id"].isin(test_vids)
    val_mask = window_meta_df["video_id"].isin(test_vids)
    
    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_va, y_va = X_all[val_mask], y_all[val_mask]
    
    if len(X_va) == 0:
        raise ValueError(f"Holdout validation set is empty. Check if TEST_VIDEO_IDS {test_vids} exist in the loaded data.")

    val_loss, model, scaler = run_training_core(
        X_tr, y_tr, X_va, y_va, cfg, label_encoder, run_dir, device, fold=1
    )
    
    if mlflow.active_run():
        mlflow.log_metric("val_loss", float(val_loss))
    print(f"✅ Holdout finished. Best Val Loss: {val_loss:.4f}")

    save_path = run_dir / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_encoder": label_encoder,
        "scaler": scaler,
        "config": OmegaConf.to_container(cfg)
    }, save_path)
    print(f"Saved best Holdout model to {save_path.resolve()}")
