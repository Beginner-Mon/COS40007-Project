"""
Airflow DAG: ML Pipeline — Automated Model Training & Promotion

This DAG automates the end-to-end ML pipeline:
  1. validate_data  — ensure training CSVs exist
  2. train_model    — run train.py with Hydra overrides
  3. evaluate_model — query MLflow for the latest run's metrics
  4. promote_model  — transition the model to Production stage in MLflow

Trigger manually from the Airflow UI (http://localhost:8080),
or set schedule_interval to "@daily" / "@weekly" for automation.

DAG Parameters (configurable via Airflow UI "Trigger DAG w/ config"):
  - task_name   : Hydra task config name (default: "activity_recognition")
  - model_type  : Model architecture (default: "bilstm")
  - epochs      : Number of training epochs (default: 50)
  - min_accuracy: Minimum avg_val_acc to promote model (default: 0.80)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# ---------------------------------------------------------------------------
# Default arguments
# ---------------------------------------------------------------------------
default_args = {
    "owner": "cos40007-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------------------------------
# DAG parameters (can be overridden at trigger-time via Airflow UI)
# ---------------------------------------------------------------------------
DAG_PARAMS = {
    "task_name": "activity_recognition",
    "model_type": "bilstm",
    "epochs": 50,
    "min_accuracy": 0.80,
}

BACKEND_DIR = "/opt/airflow/backend"
MLFLOW_DB = f"sqlite:///{BACKEND_DIR}/mlflow_tracking.db"


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------
def validate_data(**context):
    """Check that all required CSV files exist in output_data/."""
    from pathlib import Path

    data_dir = Path(BACKEND_DIR) / "output_data"
    required_files = [
        "P1_boning.csv", "P1_slicing.csv",
        "P2_boning.csv", "P2_slicing.csv",
    ]

    missing = []
    for fname in required_files:
        fpath = data_dir / fname
        if not fpath.exists():
            missing.append(str(fpath))

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} required data file(s):\n" +
            "\n".join(f"  - {m}" for m in missing)
        )

    print(f"✅ All {len(required_files)} data files found in {data_dir}")


def evaluate_model(**context):
    """
    Query MLflow for the latest run's avg_val_acc.
    Push the accuracy to XCom for the branching decision.
    """
    import mlflow

    params = context["params"]
    task_name = params.get("task_name", DAG_PARAMS["task_name"])
    min_accuracy = float(params.get("min_accuracy", DAG_PARAMS["min_accuracy"]))

    mlflow.set_tracking_uri(MLFLOW_DB)
    experiment = mlflow.get_experiment_by_name(task_name)

    if experiment is None:
        raise ValueError(f"No MLflow experiment found with name '{task_name}'")

    # Get the latest run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(f"No runs found for experiment '{task_name}'")

    latest_run = runs.iloc[0]
    avg_val_acc = latest_run.get("metrics.avg_val_acc", 0.0)
    avg_val_loss = latest_run.get("metrics.avg_val_loss", float("inf"))
    run_id = latest_run["run_id"]

    print(f"📊 Latest run: {run_id}")
    print(f"   avg_val_acc  = {avg_val_acc:.4f}")
    print(f"   avg_val_loss = {avg_val_loss:.4f}")
    print(f"   threshold    = {min_accuracy:.4f}")

    # Push metrics to XCom for downstream tasks
    context["ti"].xcom_push(key="avg_val_acc", value=float(avg_val_acc))
    context["ti"].xcom_push(key="run_id", value=run_id)

    return float(avg_val_acc)


def decide_promotion(**context):
    """Branch: promote if accuracy meets threshold, otherwise skip."""
    params = context["params"]
    min_accuracy = float(params.get("min_accuracy", DAG_PARAMS["min_accuracy"]))
    avg_val_acc = context["ti"].xcom_pull(key="avg_val_acc", task_ids="evaluate_model")

    if avg_val_acc is not None and avg_val_acc >= min_accuracy:
        print(f"✅ Accuracy {avg_val_acc:.4f} >= {min_accuracy:.4f} — promoting model.")
        return "promote_model"
    else:
        print(f"⚠️ Accuracy {avg_val_acc:.4f} < {min_accuracy:.4f} — skipping promotion.")
        return "skip_promotion"


def promote_model(**context):
    """Transition the latest model version to 'Production' in MLflow registry."""
    import mlflow
    from mlflow.tracking import MlflowClient

    params = context["params"]
    task_name = params.get("task_name", DAG_PARAMS["task_name"])

    mlflow.set_tracking_uri(MLFLOW_DB)
    client = MlflowClient()

    # Get the latest model version
    versions = client.search_model_versions(f"name='{task_name}'")
    if not versions:
        raise ValueError(f"No registered model versions found for '{task_name}'")

    latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]

    # Transition to Production
    client.transition_model_version_stage(
        name=task_name,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True,
    )

    avg_val_acc = context["ti"].xcom_pull(key="avg_val_acc", task_ids="evaluate_model")
    print(
        f"🚀 Model '{task_name}' v{latest_version.version} promoted to Production "
        f"(accuracy={avg_val_acc:.4f})"
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    description="End-to-end ML training pipeline with MLflow integration",
    schedule_interval=None,  # Manual trigger only. Change to "@daily" for automation.
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "training", "pytorch", "mlflow"],
    params=DAG_PARAMS,
) as dag:

    # Task 1: Validate data files exist
    task_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    # Task 2: Run training via Hydra CLI
    # Params are injected at trigger-time; defaults come from DAG_PARAMS.
    train_cmd = (
        f"cd {BACKEND_DIR} && "
        "python train.py "
        "task={{ params.task_name }} "
        "model.type={{ params.model_type }} "
        "train.epochs={{ params.epochs }} "
        "hydra.run.dir=outputs/airflow/${AIRFLOW_RUN_ID:-manual}"
    )

    task_train = BashOperator(
        task_id="train_model",
        bash_command=train_cmd,
    )

    # Task 3: Evaluate the latest run metrics
    task_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    # Task 4: Branch — promote or skip
    task_decide = BranchPythonOperator(
        task_id="decide_promotion",
        python_callable=decide_promotion,
    )

    # Task 5a: Promote model to Production
    task_promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    # Task 5b: Skip promotion (no-op)
    task_skip = EmptyOperator(
        task_id="skip_promotion",
    )

    # ---- DAG wiring -------------------------------------------------------
    task_validate >> task_train >> task_evaluate >> task_decide
    task_decide >> [task_promote, task_skip]
