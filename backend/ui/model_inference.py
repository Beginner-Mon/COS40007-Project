"""
Streamlit page: Model Inference

Connects to the BentoML serving endpoint to perform live predictions
on motion sensor data.
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

# Ensure project root is on path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BENTOML_URL = "http://localhost:3000"


def _check_service_health() -> bool:
    """Check if BentoML service is reachable."""
    try:
        resp = requests.post(f"{BENTOML_URL}/model_info", timeout=3)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def _get_model_info() -> dict | None:
    """Fetch model metadata from the BentoML service."""
    try:
        resp = requests.post(f"{BENTOML_URL}/model_info", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _predict(input_array: np.ndarray) -> dict | None:
    """Send a single window to the /predict endpoint."""
    try:
        resp = requests.post(
            f"{BENTOML_URL}/predict",
            json=input_array.tolist(),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Server returned status {resp.status_code}: {resp.text}")
    except requests.ConnectionError:
        st.error("Cannot connect to BentoML service. Is it running?")
    except Exception as e:
        st.error(f"Prediction request failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Page Layout
# ---------------------------------------------------------------------------
st.header("🎯 Model Inference")
st.caption("Send sensor data to the BentoML API and get real-time predictions.")

# ---- Service status indicator ---------------------------------------------
service_online = _check_service_health()
model_info = None

if service_online:
    st.success("✅ BentoML service is **online**")
    model_info = _get_model_info()
    if model_info:
        col1, col2, col3 = st.columns(3)
        col1.metric("Model", model_info.get("model_type", "?"))
        col2.metric("Task", model_info.get("task", "?"))
        col3.metric("Classes", model_info.get("num_classes", "?"))

        with st.expander("📋 Full Model Info"):
            st.json(model_info)
else:
    st.error(
        "❌ BentoML service is **offline**. Start it with:\n\n"
        "```bash\n"
        "cd backend\n"
        "bentoml serve serving.service:MotionClassifier --reload\n"
        "```"
    )

st.divider()

# ---- Input Methods --------------------------------------------------------
tab_upload, tab_random, tab_manual = st.tabs(["📁 Upload CSV", "🎲 Random Sample", "✏️ Manual JSON"])

# ---- Tab 1: Upload CSV ---------------------------------------------------
with tab_upload:
    st.markdown("Upload a CSV file containing sensor features. Each row = one timestep.")

    uploaded = st.file_uploader("Select CSV file", type=["csv"], key="inference_csv")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]} timesteps × {df.shape[1]} features")

            # Allow user to select which columns to use as features
            all_cols = df.columns.tolist()
            non_feature = ["Label", "label", "video_id", "frame", "segment", "sensor_type",
                           "activity_type", "participant", "sharpness_class", "sharpness_level",
                           "Segment Label"]
            default_features = [c for c in all_cols if c not in non_feature]
            selected_features = st.multiselect(
                "Select feature columns",
                all_cols,
                default=default_features,
                key="upload_features",
            )

            if st.button("🚀 Run Prediction", key="btn_upload") and service_online:
                input_array = df[selected_features].values.astype(np.float32)
                with st.spinner("Sending to BentoML..."):
                    result = _predict(input_array)
                if result and "error" not in result:
                    st.success(f"**Prediction:** `{result['prediction']}`  |  **Confidence:** `{result['confidence']:.2%}`")

                    # Probability bar chart
                    probs = result.get("probabilities", {})
                    if probs:
                        prob_df = pd.DataFrame(
                            {"Class": list(probs.keys()), "Probability": list(probs.values())}
                        ).sort_values("Probability", ascending=True)
                        st.bar_chart(prob_df.set_index("Class"))
                elif result:
                    st.error(result.get("error", "Unknown error"))

        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")

# ---- Tab 2: Random Sample ------------------------------------------------
with tab_random:
    st.markdown("Generate a random input array to test the service.")

    if model_info:
        n_features = model_info.get("input_features", 30)
        pad_len = model_info.get("pad_target_len", 60)
    else:
        n_features = st.number_input("Number of features", value=30, min_value=1, key="rand_features")
        pad_len = 60

    seq_len = st.slider("Sequence length", min_value=10, max_value=120, value=pad_len, key="rand_seq_len")

    if st.button("🎲 Generate & Predict", key="btn_random") and service_online:
        random_input = np.random.randn(seq_len, n_features).astype(np.float32)
        st.caption(f"Generated random array: shape ({seq_len}, {n_features})")

        with st.spinner("Sending to BentoML..."):
            result = _predict(random_input)

        if result and "error" not in result:
            st.success(f"**Prediction:** `{result['prediction']}`  |  **Confidence:** `{result['confidence']:.2%}`")

            probs = result.get("probabilities", {})
            if probs:
                prob_df = pd.DataFrame(
                    {"Class": list(probs.keys()), "Probability": list(probs.values())}
                ).sort_values("Probability", ascending=True)
                st.bar_chart(prob_df.set_index("Class"))
        elif result:
            st.error(result.get("error", "Unknown error"))

# ---- Tab 3: Manual JSON --------------------------------------------------
with tab_manual:
    st.markdown("Paste a raw JSON 2D array directly.")

    json_input = st.text_area(
        "JSON input (2D array)",
        value='[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]',
        height=150,
        key="manual_json",
    )

    if st.button("🚀 Predict from JSON", key="btn_manual") and service_online:
        try:
            parsed = np.array(json.loads(json_input), dtype=np.float32)
            if parsed.ndim != 2:
                st.error(f"Expected 2D array, got {parsed.ndim}D")
            else:
                st.caption(f"Parsed array shape: {parsed.shape}")
                with st.spinner("Sending to BentoML..."):
                    result = _predict(parsed)
                if result and "error" not in result:
                    st.success(f"**Prediction:** `{result['prediction']}`  |  **Confidence:** `{result['confidence']:.2%}`")
                    probs = result.get("probabilities", {})
                    if probs:
                        prob_df = pd.DataFrame(
                            {"Class": list(probs.keys()), "Probability": list(probs.values())}
                        ).sort_values("Probability", ascending=True)
                        st.bar_chart(prob_df.set_index("Class"))
                elif result:
                    st.error(result.get("error", "Unknown error"))
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
