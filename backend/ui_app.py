import streamlit as st
import subprocess
import sys
import os
import time
import urllib.request

st.set_page_config(page_title="Automated ML Pipeline", layout="wide")

st.title("🚀 Pipeline Manager")

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

# Hyperparameters
st.sidebar.subheader("Hyperparameters")
epochs = st.sidebar.slider("Epochs", 1, 100, 30, 1)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64, 128], index=2)
lr_str = st.sidebar.text_input("Learning Rate", "1e-4")

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
        [sys.executable, "-m", "mlflow", "ui", "--port", "5000", "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    time.sleep(2) # let it boot

is_mlflow_up = _is_mlflow_up(mlflow_url)

if is_mlflow_up:
    st.success("MLflow is running.")
    st.link_button("Open MLflow Dashboard", mlflow_url)
else:
    st.warning("MLflow is not reachable on 127.0.0.1:5000 yet. Wait a few seconds and refresh.")

st.sidebar.markdown("---")
# Training Trigger
if st.sidebar.button("🔥 Start Training Run", type="primary"):
    with st.spinner(f"Running pipeline for {selected_task}... Check terminal or MLflow for progress."):
        cmd = [
            sys.executable, "train.py",
            f"task={selected_task}",
            f"train.epochs={epochs}",
            f"data.batch_size={batch_size}",
            f"train.lr={lr_str}"
        ]
        
        if selected_strategy != "auto":
            cmd.append(f"task.split_strategy={selected_strategy}")
        
        # Run subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            st.success("Training completed successfully!")
            with st.expander("Show Logs"):
                st.code(result.stdout)
        else:
            st.error("Training failed!")
            with st.expander("Show Error Logs"):
                st.code(result.stderr)
                if result.stdout:
                    st.code(result.stdout)
