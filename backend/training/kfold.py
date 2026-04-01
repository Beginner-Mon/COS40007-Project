import numpy as np
import torch
from omegaconf import OmegaConf
import mlflow
from sklearn.model_selection import StratifiedGroupKFold

from training.core_run import run_training_core


def run_kfold_training(X_all, y_all, window_meta_df, cfg, run_dir, label_encoder, device):
    print(f"[START] Launching {cfg.task.n_splits}-Fold StratifiedGroupKFold Validation...", flush=True)
    groups = window_meta_df["video_id"].astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=cfg.task.n_splits, shuffle=True, random_state=cfg.seed)
    folds = list(cv.split(X_all, y_all, groups=groups))
    
    fold_val_losses = []
    best_model_overall = None
    best_scaler_overall = None
    best_loss_overall = float('inf')

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_va, y_va = X_all[val_idx], y_all[val_idx]
        
        val_loss, model, scaler = run_training_core(
            X_tr, y_tr, X_va, y_va, cfg, label_encoder, run_dir, device, fold=fold_idx
        )
        fold_val_losses.append(val_loss)
        
        if val_loss < best_loss_overall:
            best_loss_overall = val_loss
            best_model_overall = model
            best_scaler_overall = scaler
            
    if mlflow.active_run():
        mlflow.log_metric("avg_val_loss", float(np.mean(fold_val_losses)))
    print(f"[DONE] KFold finished. Avg Val Loss: {np.mean(fold_val_losses):.4f}", flush=True)
    
    # Save best overall
    save_path = run_dir / "best_model.pt"
    torch.save({
        "model_state_dict": best_model_overall.state_dict(),
        "label_encoder": label_encoder,
        "scaler": best_scaler_overall,
        "config": OmegaConf.to_container(cfg)
    }, save_path)
    print(f"Saved best K-Fold model to {save_path.resolve()}", flush=True)
