import streamlit as st
import subprocess
import urllib.request
import sys
import time
from ui.utils import (
    APP_DIR, 
    MLFLOW_BACKEND_URI, 
    DEFAULT_KFOLD_TASKS,
    _install_uploaded_dataset, 
    _run_streaming_command
)

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
