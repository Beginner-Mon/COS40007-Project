import streamlit as st
import subprocess
import sys
import time
import urllib.request
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Automated ML Pipeline", layout="wide")

st.title("🚀 Pipeline Manager")

APP_DIR = Path(__file__).resolve().parent
DATASET_DIR = APP_DIR / "dataset"

EPOCH_LOG_PATTERN = re.compile(r"(?:\[Fold\s+(\d+)\]\s+)?Epoch\s+(\d+)\s*/\s*(\d+)")
KFOLD_START_PATTERN = re.compile(r"\[START\]\s+Launching\s+(\d+)-Fold")
PHASE_DATA_PATTERN = re.compile(r"\[phase=data\]")
MLFLOW_RUN_PATTERN = re.compile(
    r"\[mlflow\]\s+tracking_uri=([^\s]+)\s+experiment_id=([^\s]+)\s+run_id=([a-f0-9]+)"
)

DEFAULT_KFOLD_TASKS = {"boning_vs_slicing", "knife_sharpness", "activity_recognition"}
MLFLOW_BACKEND_URI = "sqlite:///mlflow_tracking.db"


def _is_within_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_zip(uploaded_zip, extract_dir: Path) -> None:
    uploaded_zip.seek(0)
    with zipfile.ZipFile(uploaded_zip) as zf:
        for member in zf.infolist():
            member_path = extract_dir / member.filename
            if not _is_within_directory(member_path, extract_dir):
                raise ValueError("ZIP contains unsafe file paths.")
        zf.extractall(extract_dir)


def _find_dataset_root(extract_dir: Path) -> Path | None:
    candidates = [extract_dir] + [p for p in extract_dir.rglob("*") if p.is_dir()]
    for candidate in candidates:
        if (candidate / "P1").is_dir() and (candidate / "P2").is_dir():
            return candidate
    return None


def _validate_dataset_structure(dataset_root: Path) -> tuple[bool, str]:
    required_paths = [
        dataset_root / "P1" / "Boning",
        dataset_root / "P1" / "Slicing",
        dataset_root / "P2" / "Boning",
        dataset_root / "P2" / "Slicing",
    ]
    missing = [str(path.relative_to(dataset_root)) for path in required_paths if not path.is_dir()]
    if missing:
        return False, f"Missing required folders: {', '.join(missing)}"

    total_excel = sum(1 for _ in dataset_root.rglob("*.xlsx"))
    if total_excel == 0:
        return False, "No .xlsx files found in uploaded dataset."

    return True, f"Validated dataset with {total_excel} Excel files."


def _install_uploaded_dataset(uploaded_zip) -> tuple[bool, str]:
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _safe_extract_zip(uploaded_zip, tmp_path)
            dataset_root = _find_dataset_root(tmp_path)
            if dataset_root is None:
                return False, "Could not find dataset root containing P1 and P2 folders in ZIP."

            ok, msg = _validate_dataset_structure(dataset_root)
            if not ok:
                return False, msg

            if DATASET_DIR.exists():
                shutil.rmtree(DATASET_DIR)
            shutil.copytree(dataset_root, DATASET_DIR)

        return True, "Custom dataset installed successfully."
    except zipfile.BadZipFile:
        return False, "Uploaded file is not a valid ZIP archive."
    except Exception as exc:
        return False, f"Failed to install dataset: {exc}"


def _run_streaming_command(
    cmd: list[str],
    title: str,
    track_epoch_progress: bool = False,
) -> tuple[int, str]:
    st.markdown(f"### {title}")
    st.caption(f"Running command: {' '.join(cmd)}")

    logs_placeholder = st.empty()
    progress_bar = st.empty()
    status_placeholder = st.empty()
    mlflow_run_placeholder = st.empty()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(APP_DIR),
        bufsize=1,
    )

    logs: list[str] = []
    fold_count = 1
    if track_epoch_progress:
        progress_bar.progress(0.0, text="Preparing training process...")

    for line in iter(process.stdout.readline, ""):
        clean_line = line.rstrip("\n")
        logs.append(clean_line)
        # Keep full logs for post-run handling, but render only tail lines for smoother UI updates.
        logs_placeholder.code("\n".join(logs[-300:]), language="bash")
        status_placeholder.text(clean_line)

        if not track_epoch_progress:
            continue

        mlflow_match = MLFLOW_RUN_PATTERN.search(clean_line)
        if mlflow_match:
            exp_id = mlflow_match.group(2)
            run_id = mlflow_match.group(3)
            run_url = f"{mlflow_url}/#/experiments/{exp_id}/runs/{run_id}"
            mlflow_run_placeholder.markdown(
                f"MLflow run detected: [{run_id}]({run_url})"
            )

        if PHASE_DATA_PATTERN.search(clean_line):
            progress_bar.progress(0.03, text="Preparing data windows...")

        fold_start_match = KFOLD_START_PATTERN.search(clean_line)
        if fold_start_match:
            fold_count = max(1, int(fold_start_match.group(1)))
            progress_bar.progress(
                0.05,
                text=f"Training started: Fold 1/{fold_count}, waiting for first epoch...",
            )

        epoch_match = EPOCH_LOG_PATTERN.search(clean_line)
        if epoch_match:
            fold = int(epoch_match.group(1)) if epoch_match.group(1) else 1
            epoch = int(epoch_match.group(2))
            total_epochs = int(epoch_match.group(3))
            total_steps = max(1, fold_count * total_epochs)
            step = min(total_steps, (fold - 1) * total_epochs + epoch)
            progress = step / total_steps
            progress_bar.progress(
                progress,
                text=f"Training progress: Fold {fold}/{fold_count}, Epoch {epoch}/{total_epochs}",
            )

    process.stdout.close()
    return_code = process.wait()

    if track_epoch_progress:
        if return_code == 0:
            progress_bar.progress(1.0, text="Training completed.")
        else:
            progress_bar.progress(0.0, text="Training failed before completion.")

    return return_code, "\n".join(logs)

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Pipeline Architecture & Data Flow", "Data Preprocessing & EDA", "Model Training"])

if page == "Pipeline Architecture & Data Flow":
    st.header("⚙️ Pipeline Architecture & Data Flow")
    st.markdown("This page provides a visual and interactive trace of the data processing algorithms turning raw CSVs into PyTorch Tensors.")
    
    st.subheader("1. Conceptual Data Flow")
    st.markdown("""
```mermaid
flowchart LR
    classDef io fill:#f9f,stroke:#333,stroke-width:2px;
    classDef proc fill:#bbf,stroke:#333,stroke-width:2px;
    classDef model fill:#fbb,stroke:#333,stroke-width:2px;

    style DataFolder fill:#f9f,stroke:#333,stroke-width:4px
    style DataLoader fill:#fbb,stroke:#333,stroke-width:4px

    DataFolder(["output_data Folder"]):::io --> CSV["Load CSV Files"]:::proc
    CSV --> Merge["Merge Sensors"]:::proc
    Merge --> LabelClean["Label Cleaning"]:::proc
    LabelClean --> FilterLabel["Filter Excluded Labels"]:::proc
    FilterLabel --> FeatEng["Feature Engineering"]:::proc
    FeatEng --> Cleaning["Sanitize Features"]:::proc
    Cleaning --> Windowing["Create Windows"]:::proc
    Windowing --> Padding["Pad Windows"]:::proc
    Padding --> Split["Train/Validation Split"]:::proc
    Split --> Scale["Scale Normalization"]:::proc
    Scale --> Dataset["PyTorch MotionDataset"]:::proc
    Dataset --> DataLoader(["PyTorch DataLoader"]):::model
```
""")

    st.subheader("2. Live Preprocessing Trace")
    data_dir = APP_DIR / "output_data"
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            col1, col2 = st.columns(2)
            with col1:
                selected_csv = st.selectbox("Select Target Dataset (CSV)", [f.name for f in csv_files])
            
            if selected_csv:
                csv_path = data_dir / selected_csv
                with st.spinner("Discovering Video IDs..."):
                    # Fast loading just to get video_ids
                    vid_df = pd.read_csv(csv_path, usecols=['video_id'])
                    unique_vids = vid_df['video_id'].dropna().unique().tolist()
                
                with col2:
                    selected_v_id = st.selectbox("Select Video ID Trace", unique_vids)

                if st.button("Run Preprocessing Simulation", type="primary"):
                    st.session_state.show_sim = True
                    st.session_state.sim_vid = selected_v_id
                    st.session_state.sim_csv = str(csv_path)

                if st.session_state.get("show_sim", False) and st.session_state.get("sim_vid") == selected_v_id and st.session_state.get("sim_csv") == str(csv_path):
                    
                    @st.cache_data(show_spinner=False)
                    def load_and_preprocess_sample(csv_path_str, v_id):
                        import pandas as pd
                        from data.preprocessing import (
                            merge_velocity_and_acceleration,
                            clean_labels,
                            engineer_features
                        )
                        chunks = []
                        for c in pd.read_csv(csv_path_str, chunksize=50000):
                            chunks.append(c[c['video_id'] == v_id])
                        raw = pd.concat(chunks, ignore_index=True)
                        if raw.empty:
                            return None, None, None, None, None
                        
                        merged, base_cols = merge_velocity_and_acceleration(raw)
                        cleaned = clean_labels(merged)
                        eng, f_cols = engineer_features(cleaned.copy(), base_cols)
                        return raw, merged, cleaned, eng, f_cols
                        
                    with st.spinner(f"Processing '{selected_v_id}' (cached)..."):
                        raw_df, merged_df, cleaned_df, eng_df, feat_cols = load_and_preprocess_sample(str(csv_path), selected_v_id)
                    
                    if raw_df is None or raw_df.empty:
                        st.warning("No data found for this video ID.")
                    else:
                        from data.preprocessing import create_windows
                        import numpy as np
                        
                        st.markdown("---")
                        st.markdown("#### Stage 1: Raw DataFrame Loading")
                        st.markdown(f"Loaded **{len(raw_df)} rows** strictly belonging to `{selected_v_id}`. Notice the vertically staggered Velocity and Acceleration rows.")
                        st.dataframe(raw_df.head(6), use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("#### Stage 2: Sensor Alignment (`merge_velocity_and_acceleration`)")
                        st.markdown(f"Horizontally zipped by strictly matching Frame identifiers. Row count shrunk to **{len(merged_df)} pairs**. Columns suffixed with `_vel` and `_acc`.")
                        st.dataframe(merged_df.head(4), use_container_width=True)

                        st.markdown("---")
                        st.markdown("#### Stage 3: Label Cleaning (`clean_labels`)")
                        st.markdown("Mapped raw string labels into pure Integers (`int64`). Hidden bug-overrides (like Boning Label 5 -> 8) triggered here.")
                        st.write(f"Target Column Data Type: `{cleaned_df['Label'].dtype}`")
                        st.dataframe(cleaned_df[["video_id", "Frame", "Label", "activity_type"]].head(4), use_container_width=True)

                        st.markdown("---")
                        st.markdown("#### Stage 4: Feature Engineering (`engineer_features`)")
                        st.markdown(f"Mathematical feature extraction. Column count exploded from `{len(cleaned_df.columns)}` to `{len(eng_df.columns)}`.")
                        
                        st.markdown("**Core Algorithms:**")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.caption("3D Vector Magnitude")
                            st.latex(r"M = \sqrt{v_x^2 + v_y^2 + v_z^2}")
                        with c2:
                            st.caption("Total Kinetic Energy")
                            st.latex(r"E_{total} = \sum_{joint} (v^2 + a^2)")
                        with c3:
                            st.caption("Limb Energy Ratio")
                            st.latex(r"Ratio = \frac{E_{upper}}{E_{upper} + E_{lower} + 1e^{-8}}")
                            
                        # Filter out raw sensor columns to display purely engineered subsets
                        engineered_cols_only = [col for col in eng_df.columns if not col.endswith('_vel') and not col.endswith('_acc')]
                        display_cols = ["video_id", "Frame", "Label"] + [c for c in engineered_cols_only if c not in ["video_id", "Frame", "Label", "activity_type"]]
                        st.dataframe(eng_df[display_cols].head(4), use_container_width=True)

                        st.markdown("---")
                        st.markdown("#### Stage 5: Target Windowing (`create_windows`)")
                        
                        target_col_selection = st.selectbox("Select Target Sequence Boundary", ["Label", "activity_type", "sharpness_class", "knife_sharpness_score"])
                        
                        # Derive sharpness_class if needed
                        if target_col_selection == "sharpness_class" and "sharpness_class" not in eng_df.columns:
                            from data.preprocessing import derive_sharpness_class
                            eng_df = derive_sharpness_class(eng_df)
                            
                        st.markdown(f"Constructing sliding time-series windows (hardcoded `window_size=60`, `stride=30`, `frame_skip_step=2`). Sequence terminates inherently if **`{target_col_selection}`** changes or Frame jumps > 1.")
                        
                        try:
                            # Use default parameters as requested
                            X_windows, y_windows, window_meta_df = create_windows(
                                eng_df, 
                                feat_cols, 
                                window_size=60, 
                                stride=30, 
                                target_col=target_col_selection, 
                                frame_skip_step=2
                            )
                            
                            if len(X_windows) > 0:
                                sample_shape = X_windows[0].shape
                                st.success(f"Successfully generated **{len(X_windows)} sequences**. Output Tensor Shape: `[{len(X_windows)}, {sample_shape[0]}, {sample_shape[1]}]`")
                                
                                st.markdown(f"**Generated `{target_col_selection}` Distribution (Post-Windowing)**")
                                dist_series = pd.Series(y_windows).value_counts().sort_index()
                                # Convert series index to string names for better readability on chart
                                dist_series.index = dist_series.index.map(lambda x: f"{target_col_selection}: {x}")
                                st.bar_chart(dist_series)
                            else:
                                st.warning("Not enough continuous frames to build a 60-frame window from this subset.")
                        except Exception as e:
                            st.error(f"Windowing failed: {e}")

elif page == "Data Preprocessing & EDA":
    st.header("Data Preprocessing & EDA")
    data_dir = APP_DIR / "output_data"
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            selected_csv = st.selectbox("Select Dataset to Explore", [f.name for f in csv_files])
            if selected_csv:
                st.write(f"### Loading `{selected_csv}`...")
                
                load_strategy = st.radio("Data Load Strategy", ["Load Top N Rows", "Load Specific Video ID"], horizontal=True)
                
                try:
                    if load_strategy == "Load Top N Rows":
                        row_limit_opt = st.selectbox("Max Rows to Load", ["1,000", "10,000", "50,000", "100,000", "All"], index=1)
                        nrows = None if row_limit_opt == "All" else int(row_limit_opt.replace(",", ""))
                        with st.spinner("Loading rows..."):
                            df = pd.read_csv(data_dir / selected_csv, nrows=nrows, low_memory=False)
                    else:
                        with st.spinner("Scanning for available Video IDs..."):
                            vid_scan_df = pd.read_csv(data_dir / selected_csv, usecols=['video_id'], dtype=str)
                            available_vids = vid_scan_df['video_id'].dropna().unique().tolist()
                            
                        selected_video_id = st.selectbox("Select Target Video ID to load", available_vids)
                        
                        with st.spinner(f"Loading data exclusively for Video {selected_video_id}..."):
                            # Read in chunks to efficiently filter heavily weighted files
                            chunk_iter = pd.read_csv(data_dir / selected_csv, chunksize=100000, low_memory=False)
                            df_list = []
                            for chunk in chunk_iter:
                                df_list.append(chunk[chunk['video_id'].astype(str) == selected_video_id])
                            df = pd.concat(df_list, ignore_index=True)
                            
                            # Ensure clean line plots by sorting sequentially by frame
                            if 'Frame' in df.columns:
                                df = df.sort_values(by="Frame").reset_index(drop=True)
                            
                    st.write(f"**Data Preview ({len(df)} rows)**")
                    st.dataframe(df)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Summary Statistics**")
                        st.dataframe(df.describe())
                    with col2:
                        st.write("**Missing Values**")
                        st.dataframe(df.isna().sum())

                    st.markdown("---")
                    st.write("### 📈 Visual Exploratory Data Analysis")
                    
                    excluded_cols = ['Frame', 'person_id', 'activity_type', 'knife_sharpness_score', 'sensor_type', 'video_id', 'Label']
                    feature_cols = [c for c in df.columns if c not in excluded_cols and pd.api.types.is_numeric_dtype(df[c])]
                    
                    if not feature_cols:
                        st.warning("Not enough numeric features for advanced EDA plotting.")
                    else:
                        selected_feature = st.selectbox("Select Feature for Visualizations", feature_cols, index=0)
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            st.write("**Histogram & Distribution Analysis**")
                            fig_hist = px.histogram(df, x=selected_feature, marginal="box", title=f"Distribution of {selected_feature}")
                            st.plotly_chart(fig_hist, width="stretch")
                            
                        with viz_col2:
                            st.write("**Correlation Heatmap**")
                            st.caption("Excluding specific metadata columns")
                            corr_matrix = df[feature_cols].corr()
                            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax_corr, square=True)
                            st.pyplot(fig_corr)
                            
                        st.markdown("---")
                        st.write("**Time Series Line Plot (by Frame)**")
                        x_col = 'Frame' if 'Frame' in df.columns else df.index
                        
                        # Fix jagged lines: Group colors so Plotly doesn't connect different sensors/videos at the same Frame.
                        color_col = None
                        if load_strategy == "Load Top N Rows":
                            if 'video_id' in df.columns and 'sensor_type' in df.columns:
                                df['_plot_group'] = df['video_id'].astype(str) + " | " + df['sensor_type'].astype(str)
                                color_col = '_plot_group'
                            elif 'video_id' in df.columns:
                                color_col = 'video_id'
                            elif 'sensor_type' in df.columns:
                                color_col = 'sensor_type'
                        else:
                            if 'sensor_type' in df.columns:
                                color_col = 'sensor_type'
                                
                        fig_line = px.line(df, x=x_col, y=selected_feature, color=color_col, title=f"{selected_feature} Tracking over {x_col}")
                        st.plotly_chart(fig_line, width="stretch")
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
        else:
            st.info("No CSV files found in output_data folder.")
    else:
        st.warning("outputs_data folder does not exist.")

elif page == "Model Training":
    st.sidebar.header("Configuration")

    # Task Selection
    task_map = {
        "Boning vs Slicing": "boning_vs_slicing",
        "Knife Sharpness (3-Class)": "knife_sharpness",
        "Activity Recognition (Full)": "activity_recognition"
    }
    selected_task_label = st.sidebar.selectbox("Select Target Pipeline Task", list(task_map.keys()))
    selected_task = task_map[selected_task_label]

    # Validation Strategy Selection
    strategy_map = {
        "Auto (Use Config Default)": "auto",
        "Holdout (Train/Test Split)": "holdout",
        "K-Fold Cross Validation": "kfold"
    }
    selected_strategy_label = st.sidebar.selectbox("Validation Strategy", list(strategy_map.keys()))
    selected_strategy = strategy_map[selected_strategy_label]

    if selected_strategy == "auto" and selected_task in DEFAULT_KFOLD_TASKS:
        st.sidebar.info(
            "Auto strategy uses 10-fold validation for this task by default. "
            "With epochs=1, training still runs across all folds."
        )
    if selected_strategy == "holdout":
        st.sidebar.info(
            "Holdout uses task.test_video_ids when available. If empty, an automatic "
            "video_id group split is used based on data.val_split."
        )

    kfold_splits_override: int | None = None
    if selected_strategy == "kfold":
        kfold_splits_override = st.sidebar.slider("K-Fold Splits", 2, 10, 10, 1)
    elif selected_strategy == "auto" and selected_task in DEFAULT_KFOLD_TASKS:
        kfold_splits_override = st.sidebar.slider("K-Fold Splits Override", 2, 10, 10, 1)

    # Hyperparameters
    st.sidebar.subheader("Hyperparameters")
    selected_model_arch = st.sidebar.selectbox("Model Architecture", ["bilstm", "gru", "tcn"])
    epochs = st.sidebar.slider("Epochs", 1, 100, 30, 1)
    batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64, 128], index=2)
    lr_str = st.sidebar.text_input("Learning Rate", "1e-4")

    st.sidebar.subheader("Custom Dataset (Optional)")
    uploaded_dataset_zip = st.sidebar.file_uploader(
        "Upload dataset ZIP (must include P1/P2 and Boning/Slicing folders)",
        type=["zip"],
    )

    # MLflow Dashboard
    st.subheader("📊 MLflow Tracking")
    st.write("Open MLflow in a new tab to view realtime training progress, parameters, and loss curves.")

    mlflow_url = "http://127.0.0.1:5000"

    def _is_mlflow_up(url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    # Launch MLflow if not running
    if 'mlflow_proc' not in st.session_state and not _is_mlflow_up(mlflow_url):
        st.session_state.mlflow_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                MLFLOW_BACKEND_URI,
                "--port",
                "5000",
                "--host",
                "127.0.0.1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(APP_DIR)
        )
        time.sleep(2) # let it boot

    is_mlflow_up = _is_mlflow_up(mlflow_url)

    status_col, refresh_col = st.columns([0.8, 0.2])
    with refresh_col:
        st.button("🔄 Refresh Status")

    with status_col:
        if is_mlflow_up:
            st.success("MLflow is running.")
        else:
            st.warning("MLflow is not reachable on 127.0.0.1:5000 yet. Wait a few seconds and refresh.")

    if is_mlflow_up:
        mlflow_col1, mlflow_col2 = st.columns(2)
        with mlflow_col1:
            st.link_button("Open MLflow Experiments", f"{mlflow_url}/#/experiments")
        with mlflow_col2:
            st.link_button("Open MLflow Home", mlflow_url)
        st.caption("If MLflow opens in GenAI mode, switch to Model training to see training runs.")

    st.sidebar.markdown("---")
    # Training Trigger
    if st.sidebar.button("🔥 Start Training Run", type="primary"):
        if uploaded_dataset_zip is not None:
            st.info("Installing uploaded custom dataset...")
            ok, message = _install_uploaded_dataset(uploaded_dataset_zip)
            if not ok:
                st.error(message)
                st.stop()
            st.success(message)

            loader_cmd = [sys.executable, "-u", "-m", "data.loader"]
            loader_rc, _ = _run_streaming_command(loader_cmd, "Dataset Preprocessing Logs")
            if loader_rc != 0:
                st.error("Dataset preprocessing failed. Training was not started.")
                st.stop()
            st.success("Dataset preprocessing completed.")

        st.info(f"Running pipeline for {selected_task}...")
        train_cmd = [
            sys.executable,
            "-u",
            "train.py",
            f"task={selected_task}",
            f"+model.type={selected_model_arch}",
            f"train.epochs={epochs}",
            f"data.batch_size={batch_size}",
            f"train.lr={lr_str}",
        ]

        if selected_strategy != "auto":
            train_cmd.append(f"task.split_strategy={selected_strategy}")
        if kfold_splits_override is not None:
            train_cmd.append(f"task.n_splits={kfold_splits_override}")

        train_rc, _ = _run_streaming_command(
            train_cmd,
            "Training Logs (Live)",
            track_epoch_progress=True,
        )

        if train_rc == 0:
            st.success("Training completed successfully!")
        else:
            st.error("Training failed. Review logs above.")
