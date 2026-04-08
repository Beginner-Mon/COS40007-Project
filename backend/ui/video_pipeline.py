"""
Streamlit page: Video Inference Pipeline

Upload a video and visualise every stage of the pipeline:
    Stage 1 — MediaPipe 2D pose extraction
    Stage 2 — VideoPose3D 2D → 3D lifting
    Stage 3 — Joint mapping (H3.6M → Xsens)
    Stage 4 — Feature engineering
    Stage 5 — Sliding-window creation
    Stage 6 — BiLSTM prediction
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import tempfile
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.header("🎬 Video Inference Pipeline")
st.caption(
    "Upload a video and watch the pipeline process it through every stage — "
    "from raw pixels to activity predictions."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Pipeline Settings")
window_size = st.sidebar.slider("Window size", 20, 120, 60, key="vp_ws")
stride = st.sidebar.slider("Stride", 5, 60, 30, key="vp_stride")
frame_skip_step = st.sidebar.slider("Frame skip step", 1, 5, 2, key="vp_fss")
pad_target_len = st.sidebar.number_input("Pad target length", value=60, min_value=10, key="vp_pad")
mediapipe_complexity = st.sidebar.selectbox(
    "MediaPipe complexity", [0, 1, 2], index=2,
    help="0=lite, 1=full, 2=heavy (most accurate)",
    key="vp_mp_cx",
)

# ---------------------------------------------------------------------------
# Video upload
# ---------------------------------------------------------------------------
uploaded_video = st.file_uploader(
    "Upload a video file", type=["mp4", "avi", "mov", "mkv"], key="video_upload"
)

if uploaded_video is None:
    st.info("👆 Upload a video to start the pipeline.")
    st.stop()

# Show the uploaded video
st.video(uploaded_video)

# Save to a temp file so OpenCV/MediaPipe can read it
tmp_dir = PROJECT_ROOT / "_tmp_video"
tmp_dir.mkdir(exist_ok=True)
tmp_path = tmp_dir / uploaded_video.name
with open(tmp_path, "wb") as f:
    f.write(uploaded_video.getbuffer())

# ---------------------------------------------------------------------------
# Run pipeline button
# ---------------------------------------------------------------------------
if not st.button("🚀 Run Full Pipeline", type="primary", key="run_pipeline"):
    st.stop()


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1 — MediaPipe 2D Pose Extraction
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Stage 1 — MediaPipe 2D Pose Extraction")

with st.spinner("Extracting 2D poses with MediaPipe..."):
    from pipeline.video_to_2d import extract_2d_poses

    keypoints_2d, video_meta = extract_2d_poses(
        str(tmp_path),
        model_complexity=mediapipe_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Frames", video_meta["total_frames"])
col2.metric("Detected Frames", video_meta["detected_frames"])
col3.metric("Detection Rate", f"{video_meta['detected_frames'] / max(video_meta['total_frames'], 1) * 100:.1f}%")
col4.metric("FPS", f"{video_meta['fps']:.1f}")

# Visualise: plot a few 2D skeletons
with st.expander("🦴 2D Skeleton Preview (sampled frames)", expanded=True):
    n_frames = keypoints_2d.shape[0]
    sample_indices = np.linspace(0, n_frames - 1, min(6, n_frames), dtype=int)

    # H3.6M-style limb connections using MediaPipe indices
    MP_CONNECTIONS = [
        (11, 13), (13, 15), (12, 14), (14, 16),  # arms
        (11, 12), (11, 23), (12, 24), (23, 24),   # torso
        (23, 25), (25, 27), (24, 26), (26, 28),   # legs
        (0, 11), (0, 12),                          # head-shoulder
    ]

    fig_skel = go.Figure()
    offset = 0
    for idx in sample_indices:
        kp = keypoints_2d[idx]
        if np.isnan(kp).all():
            continue

        x = kp[:, 0] + offset
        y = -kp[:, 1]  # flip y for display

        # Draw limbs
        for a, b in MP_CONNECTIONS:
            if not (np.isnan(kp[a]).any() or np.isnan(kp[b]).any()):
                fig_skel.add_trace(go.Scatter(
                    x=[x[a], x[b]], y=[y[a], y[b]],
                    mode="lines", line=dict(color="rgba(0,150,255,0.5)", width=2),
                    showlegend=False, hoverinfo="skip",
                ))

        # Draw joints
        fig_skel.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=4, color="cyan"),
            name=f"Frame {idx}",
            hoverinfo="text",
            text=[f"lm {i}" for i in range(33)],
        ))
        offset += 1.5

    fig_skel.update_layout(
        title="Sampled 2D Skeletons (MediaPipe 33 landmarks)",
        xaxis=dict(visible=False), yaxis=dict(scaleanchor="x", visible=False),
        height=350, margin=dict(t=40, b=10, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_skel, use_container_width=True)

# Landmark confidence over time
with st.expander("📊 Landmark Visibility Over Time"):
    avg_vis = np.nanmean(keypoints_2d[:, :, 3], axis=1)  # avg visibility per frame
    vis_df = pd.DataFrame({"Frame": np.arange(len(avg_vis)), "Avg Visibility": avg_vis})
    fig_vis = px.line(vis_df, x="Frame", y="Avg Visibility",
                      title="Average Landmark Visibility Score per Frame")
    fig_vis.update_layout(height=250)
    st.plotly_chart(fig_vis, use_container_width=True)

if video_meta["detected_frames"] == 0:
    st.error("❌ No poses detected. Cannot continue pipeline.")
    st.stop()

st.success(f"✅ Stage 1 complete — extracted {keypoints_2d.shape} keypoints")


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2 — VideoPose3D 2D → 3D Lifting
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Stage 2 — VideoPose3D 2D → 3D Lifting")

with st.spinner("Lifting 2D poses to 3D with VideoPose3D..."):
    from pipeline.pose_2d_to_3d import lift_to_3d, mediapipe_to_h36m

    joints_3d = lift_to_3d(keypoints_2d)

col1, col2 = st.columns(2)
col1.metric("Output Shape", f"{joints_3d.shape}")
col2.metric("Joints", "17 (Human3.6M)")

# 3D skeleton visualisation
with st.expander("🦴 3D Skeleton Preview (sampled frames)", expanded=True):
    H36M_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3),     # right leg
        (0, 4), (4, 5), (5, 6),     # left leg
        (0, 7), (7, 8), (8, 9),     # spine
        (9, 10),                      # head
        (8, 14), (14, 15), (15, 16), # right arm
        (8, 11), (11, 12), (12, 13), # left arm
    ]

    H36M_JOINT_NAMES = [
        "Hip", "R.Hip", "R.Knee", "R.Ankle",
        "L.Hip", "L.Knee", "L.Ankle",
        "Spine", "Thorax", "Neck", "Head",
        "L.Shoulder", "L.Elbow", "L.Wrist",
        "R.Shoulder", "R.Elbow", "R.Wrist",
    ]

    sample_3d_idx = np.linspace(0, joints_3d.shape[0] - 1, min(4, joints_3d.shape[0]), dtype=int)
    frame_selector = st.selectbox(
        "Select frame to view in 3D",
        sample_3d_idx.tolist(),
        format_func=lambda x: f"Frame {x}",
        key="3d_frame_sel",
    )

    j = joints_3d[frame_selector]
    fig_3d = go.Figure()

    # Limbs
    for a, b in H36M_CONNECTIONS:
        fig_3d.add_trace(go.Scatter3d(
            x=[j[a, 0], j[b, 0]], y=[j[a, 1], j[b, 1]], z=[j[a, 2], j[b, 2]],
            mode="lines", line=dict(color="dodgerblue", width=5),
            showlegend=False, hoverinfo="skip",
        ))

    # Joints
    fig_3d.add_trace(go.Scatter3d(
        x=j[:, 0], y=j[:, 1], z=j[:, 2],
        mode="markers+text",
        marker=dict(size=5, color="cyan", symbol="circle"),
        text=H36M_JOINT_NAMES,
        textposition="top center",
        textfont=dict(size=8),
        name="Joints",
    ))

    fig_3d.update_layout(
        title=f"3D Skeleton — Frame {frame_selector}",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        height=500, margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# Joint trajectories
with st.expander("📈 3D Joint Trajectory Over Time"):
    joint_sel = st.selectbox("Joint", H36M_JOINT_NAMES, index=16, key="traj_joint")
    j_idx = H36M_JOINT_NAMES.index(joint_sel)
    traj_df = pd.DataFrame({
        "Frame": np.arange(joints_3d.shape[0]),
        "X": joints_3d[:, j_idx, 0],
        "Y": joints_3d[:, j_idx, 1],
        "Z": joints_3d[:, j_idx, 2],
    })
    fig_traj = px.line(traj_df, x="Frame", y=["X", "Y", "Z"],
                       title=f"{joint_sel} — 3D Position Over Time")
    fig_traj.update_layout(height=300)
    st.plotly_chart(fig_traj, use_container_width=True)

st.success(f"✅ Stage 2 complete — lifted to {joints_3d.shape}")


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3 — Joint Mapping (H3.6M → Xsens)
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Stage 3 — Joint Mapping (H3.6M → Xsens 23 Segments)")

with st.spinner("Mapping joints to Xsens segment format..."):
    from pipeline.joint_mapper import map_h36m_to_xsens, XSENS_SEGMENTS

    df_xsens = map_h36m_to_xsens(joints_3d)
    df_xsens["video_id"] = Path(uploaded_video.name).stem
    df_xsens["sensor_type"] = "Segment Position"
    df_xsens["Label"] = 0

col1, col2 = st.columns(2)
col1.metric("Rows", f"{len(df_xsens):,}")
col2.metric("Columns", f"{len(df_xsens.columns)}")

with st.expander("📋 Mapped DataFrame Preview", expanded=True):
    st.dataframe(df_xsens.head(20), use_container_width=True)

with st.expander("🗺️ Joint Mapping Table"):
    from pipeline.joint_mapper import _SEGMENT_RULES
    mapping_rows = []
    for seg, (rule_type, indices) in _SEGMENT_RULES.items():
        mapping_rows.append({
            "Xsens Segment": seg,
            "Strategy": rule_type.capitalize(),
            "H3.6M Joint Index": ", ".join(str(i) for i in indices),
        })
    st.table(pd.DataFrame(mapping_rows))

st.success(f"✅ Stage 3 complete — {len(XSENS_SEGMENTS)} segments × 3 axes = {len(XSENS_SEGMENTS) * 3} features")


# ═══════════════════════════════════════════════════════════════════════
# STAGE 4 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Stage 4 — Sensor Merge & Feature Engineering")

with st.spinner("Merging sensors and engineering features..."):
    from data.preprocessing import merge_sensors, engineer_features

    merged_df, base_feature_cols = merge_sensors(df_xsens, ["Segment Position"])
    engineered_df, feature_cols = engineer_features(merged_df, base_feature_cols)

col1, col2, col3 = st.columns(3)
col1.metric("Base Features", len(base_feature_cols))
col2.metric("Engineered Features", len(feature_cols))
col3.metric("Total Columns", len(engineered_df.columns))

with st.expander("📊 Engineered Feature Names", expanded=False):
    feat_df = pd.DataFrame({
        "Feature": feature_cols,
        "Type": [
            "Magnitude" if "_mag" in f
            else "Kinematic Pair" if any(s in f for s in ["_xy", "_xz", "_yz"])
            else "Energy" if "energy" in f.lower()
            else "Ratio" if "ratio" in f.lower()
            else "Other"
            for f in feature_cols
        ]
    })
    st.dataframe(feat_df, use_container_width=True)

    # Feature type summary
    type_counts = feat_df["Type"].value_counts()
    fig_types = px.pie(values=type_counts.values, names=type_counts.index,
                       title="Feature Type Distribution")
    st.plotly_chart(fig_types, use_container_width=True)

# Sample feature time-series
with st.expander("📈 Engineered Feature Time Series"):
    eng_feat_sel = st.selectbox("Select feature", feature_cols, index=0, key="eng_feat")
    if eng_feat_sel in engineered_df.columns:
        fig_eng = px.line(
            engineered_df, x="Frame" if "Frame" in engineered_df.columns else engineered_df.index,
            y=eng_feat_sel, title=f"{eng_feat_sel} Over Time"
        )
        fig_eng.update_layout(height=300)
        st.plotly_chart(fig_eng, use_container_width=True)

st.success(f"✅ Stage 4 complete — {len(feature_cols)} engineered features")


# ═══════════════════════════════════════════════════════════════════════
# STAGE 5 — Sliding Window Creation
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Stage 5 — Sliding Window Creation")

with st.spinner("Creating sliding windows..."):
    from data.preprocessing import create_windows, pad_windows_to_60, clean_features

    X_windows, y_windows, window_meta_df = create_windows(
        engineered_df, feature_cols, window_size, stride,
        target_col="Label", frame_skip_step=frame_skip_step,
    )

    if len(X_windows) == 0:
        st.error("❌ Video too short to create any windows with current settings.")
        st.stop()

    X_all = pad_windows_to_60(X_windows, target_len=pad_target_len).astype(np.float32)
    X_all = clean_features(X_all)

col1, col2, col3 = st.columns(3)
col1.metric("Windows Created", X_all.shape[0])
col2.metric("Timesteps/Window", X_all.shape[1])
col3.metric("Features/Timestep", X_all.shape[2])

with st.expander("📋 Window Metadata", expanded=False):
    st.dataframe(window_meta_df, use_container_width=True)

# Heatmap of a sample window
with st.expander("🔥 Sample Window Heatmap", expanded=True):
    win_idx = st.slider("Window index", 0, X_all.shape[0] - 1, 0, key="win_hm_idx")
    fig_hm = px.imshow(
        X_all[win_idx].T,
        labels=dict(x="Timestep", y="Feature", color="Value"),
        title=f"Window {win_idx} — Feature Heatmap",
        aspect="auto",
        color_continuous_scale="RdBu_r",
    )
    fig_hm.update_layout(height=400)
    st.plotly_chart(fig_hm, use_container_width=True)

st.success(f"✅ Stage 5 complete — {X_all.shape[0]} windows of shape {X_all.shape[1:]}")


# ═══════════════════════════════════════════════════════════════════════
# STAGE 6 — BiLSTM Prediction
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Stage 6 — BiLSTM Prediction")

with st.spinner("Loading model and running inference..."):
    import torch
    import torch.nn.functional as F
    from pipeline.video_inference import _load_bilstm_checkpoint, _auto_discover_checkpoint

    try:
        ckpt_path = _auto_discover_checkpoint("activity_recognition")
        bundle = _load_bilstm_checkpoint(ckpt_path)
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()

    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]

    # Validate feature count
    expected_features = scaler.n_features_in_
    actual_features = X_all.shape[2]

    if actual_features != expected_features:
        st.error(
            f"❌ Feature mismatch: model expects **{expected_features}** features "
            f"but the pipeline produced **{actual_features}**. "
            f"Ensure the model was trained with `sensor_types: ['Segment Position']`."
        )
        st.stop()

    # Scale
    B, T, Feat = X_all.shape
    X_scaled = scaler.transform(X_all.reshape(-1, Feat)).reshape(B, T, Feat)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    predictions = []
    with torch.no_grad():
        for start in range(0, B, 64):
            end = min(start + 64, B)
            logits = model(tensor[start:end])
            probs = F.softmax(logits, dim=1).cpu().numpy()
            for i in range(probs.shape[0]):
                pred_idx = int(np.argmax(probs[i]))
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
                predictions.append({
                    "window": start + i,
                    "label": str(pred_label),
                    "confidence": float(probs[i][pred_idx]),
                    "probabilities": {
                        str(c): float(p) for c, p in zip(label_encoder.classes_, probs[i])
                    },
                })

# Results summary
st.markdown("### 📊 Prediction Results")

pred_df = pd.DataFrame(predictions)
summary = pred_df["label"].value_counts()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Per-Window Predictions**")
    display_df = pred_df[["window", "label", "confidence"]].copy()
    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.2%}")
    st.dataframe(display_df, use_container_width=True, height=350)

with col2:
    st.markdown("**Class Distribution**")
    fig_pie = px.pie(
        values=summary.values, names=summary.index,
        title="Predicted Activity Distribution",
        hole=0.4,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

# Confidence over time
st.markdown("**Prediction Confidence Over Windows**")
fig_conf = px.bar(
    pred_df, x="window", y="confidence", color="label",
    title="Per-Window Confidence",
    labels={"window": "Window Index", "confidence": "Confidence"},
)
fig_conf.update_layout(height=350)
st.plotly_chart(fig_conf, use_container_width=True)

# Probability heatmap
st.markdown("**Class Probability Heatmap**")
prob_matrix = np.array([
    [p["probabilities"].get(str(c), 0) for c in label_encoder.classes_]
    for p in predictions
])
fig_prob_hm = px.imshow(
    prob_matrix.T,
    labels=dict(x="Window", y="Class", color="Probability"),
    y=[str(c) for c in label_encoder.classes_],
    title="Class Probabilities per Window",
    aspect="auto",
    color_continuous_scale="Viridis",
)
fig_prob_hm.update_layout(height=300)
st.plotly_chart(fig_prob_hm, use_container_width=True)

# Download results
result_json = json.dumps({
    "predictions": predictions,
    "summary": {str(k): int(v) for k, v in summary.items()},
    "metadata": video_meta,
    "pipeline_config": {
        "window_size": window_size, "stride": stride,
        "frame_skip_step": frame_skip_step, "pad_target_len": pad_target_len,
    },
}, indent=2)

st.download_button(
    "⬇️ Download Results JSON",
    data=result_json,
    file_name=f"{Path(uploaded_video.name).stem}_predictions.json",
    mime="application/json",
    key="dl_results",
)

st.success("✅ Pipeline complete!")

# Cleanup
try:
    tmp_path.unlink(missing_ok=True)
except Exception:
    pass
