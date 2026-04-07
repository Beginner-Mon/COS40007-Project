import streamlit as st
import subprocess
import sys
import time
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

# APP_DIR points to the backend directory (parent of ui directory)
APP_DIR = Path(__file__).resolve().parent.parent
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
    mlflow_url: str = "http://127.0.0.1:5000",
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
