import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import mlflow
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig

from utils.seed import set_seed
from utils.device import get_device

from data.preprocessing import (
    get_feature_columns,
    create_windows,
    clean_features,
    merge_velocity_and_acceleration,
    engineer_features,
    pad_windows_to_60,
    derive_sharpness_class,
    clean_labels,
)

# Import our newly decoupled modules
from training.kfold import run_kfold_training
from training.holdout import run_holdout_training

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    with open(run_dir / "config_used.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    device = get_device(cfg.device)

    PROJECT_ROOT = Path(get_original_cwd())
    DATA_DIR = PROJECT_ROOT / "output_data"

    # MLflow Setup
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_name = cfg.task.name
    mlflow.set_experiment(experiment_name)

    # Start run early so failed attempts are still visible in MLflow UI.
    with mlflow.start_run() as active_run:
        exp = mlflow.get_experiment_by_name(experiment_name)
        exp_id = exp.experiment_id if exp is not None else "unknown"
        print(
            f"[mlflow] tracking_uri={cfg.mlflow.tracking_uri} "
            f"experiment_id={exp_id} run_id={active_run.info.run_id}",
            flush=True,
        )
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # ======================================================
        # Load data (P1 + P2 combined)
        # ======================================================
        print("[phase=data] Loading CSV files...", flush=True)
        dfs = []
        for pid in ["P1", "P2"]:
            for act in ["boning", "slicing"]:
                csv_path = DATA_DIR / f"{pid}_{act}.csv"
                if csv_path.exists():
                    dfs.append(pd.read_csv(csv_path, low_memory=False))

        if not dfs:
            raise FileNotFoundError(f"No CSV files found in {DATA_DIR}. Run `python -m data.loader` first.")
        
        raw_df = pd.concat(dfs, ignore_index=True)

        # 1. Merge Sensors
        merged_df, base_feature_cols = merge_velocity_and_acceleration(raw_df)

        # 2. Map and Filter Labels
        merged_df = clean_labels(merged_df)

        excluded = list(cfg.task.get("excluded_labels", []))
        if excluded:
            merged_df = merged_df[~merged_df["Label"].isin(excluded)].copy()

        # Derive sharpness_class if needed by this task
        TARGET_COL = cfg.task.target_col
        if TARGET_COL == "sharpness_class" and "sharpness_class" not in merged_df.columns:
            merged_df = derive_sharpness_class(merged_df)

        # Dynamic target col
        if TARGET_COL in merged_df.columns and pd.api.types.is_string_dtype(merged_df[TARGET_COL]):
            merged_df[TARGET_COL] = merged_df[TARGET_COL].astype(str).str.strip().str.lower()
        
        if TARGET_COL not in merged_df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe. Available columns: {list(merged_df.columns)}")

        # 3. Engineer Features
        merged_df, feature_cols = engineer_features(merged_df, base_feature_cols)

        # Optional per-label/target dynamic window rules.
        label_specific_rules = {}
        task_windowing_cfg = cfg.task.get("windowing")
        if task_windowing_cfg:
            rules_cfg = task_windowing_cfg.get("label_specific_rules", {})
            if rules_cfg:
                label_specific_rules = OmegaConf.to_container(rules_cfg, resolve=True)

        # 4. Create Windows
        X_windows, y_windows, window_meta_df = create_windows(
            merged_df, 
            feature_cols, 
            cfg.data.window_size, 
            cfg.data.stride,
            TARGET_COL,
            frame_skip_step=cfg.data.get("frame_skip_step", 2),
            label_specific_rules=label_specific_rules,
        )

        X_all = pad_windows_to_60(
            X_windows,
            target_len=int(cfg.data.get("pad_target_len", 60)),
        ).astype(np.float32)
        y_windows = np.array(y_windows, dtype=object)

        label_encoder = LabelEncoder()
        y_all = label_encoder.fit_transform(y_windows).astype(np.int64)

        # Clean Features
        X_all = clean_features(X_all)

        print(f"[phase=data] Total Windows Generated: {len(X_all)}", flush=True)
        if not window_meta_df.empty and {"target", "window_size", "window_stride", "frame_skip_step"}.issubset(window_meta_df.columns):
            print("[phase=data] effective window settings by target:", flush=True)
            settings_by_target = (
                window_meta_df[["target", "window_size", "window_stride", "frame_skip_step"]]
                .drop_duplicates()
                .sort_values("target")
            )
            print(settings_by_target.to_string(index=False), flush=True)

        # ======================================================
        # Delegate to Specialized Validation Strategy Modules
        # ======================================================
        if cfg.task.split_strategy == "kfold":
            run_kfold_training(X_all, y_all, window_meta_df, cfg, run_dir, label_encoder, device)
            
        elif cfg.task.split_strategy == "holdout":
            run_holdout_training(X_all, y_all, window_meta_df, cfg, run_dir, label_encoder, device)
            
        else:
            raise ValueError(f"Unknown split strategy: {cfg.task.split_strategy}")
            
if __name__ == "__main__":
    main()
