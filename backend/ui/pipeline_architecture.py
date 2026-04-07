import streamlit as st
import pandas as pd
from ui.utils import APP_DIR

st.header("⚙️ Pipeline Architecture & Data Flow")
st.markdown("This page provides a visual and interactive trace of the data processing algorithms turning raw CSVs into PyTorch Tensors.")

st.subheader("1. Conceptual Data Flow")
st.graphviz_chart('''
digraph DataFlow {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="sans-serif", fillcolor="#e0e5ff", color="#a3b1ff", margin=0.2];
    edge [color="#8c9eff", penwidth=2];
    
    Source [label="output_data Folder", shape=cylinder, fillcolor="#ffccbc", color="#ffab91"];
    Target [label="PyTorch DataLoader", shape=cylinder, fillcolor="#c8e6c9", color="#a5d6a7"];
    
    Source -> "Load CSV" -> "Merge Sensors" -> "Label Cleaning" -> "Filter Labels" -> "Engineer Features" -> "Sanitize Features" -> "Create Windows" -> "Pad Windows" -> "Train / Val Split" -> "Standard Scaler" -> Target;
}
''')

st.subheader("2. Live Preprocessing Trace")
data_dir = APP_DIR / "output_data"
if data_dir.exists():
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        col1, col2, col3 = st.columns(3)
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
            
            with col3:
                task_yaml_dir = APP_DIR / "configs/task"
                task_files = [f.stem for f in task_yaml_dir.glob("*.yaml")] if task_yaml_dir.exists() else ["activity_recognition", "boning_vs_slicing", "knife_sharpness"]
                selected_task = st.selectbox("Select ML Task Config", task_files)

            if st.button("Run Preprocessing Simulation", type="primary"):
                st.session_state.show_sim = True
                st.session_state.sim_vid = selected_v_id
                st.session_state.sim_csv = str(csv_path)
                st.session_state.sim_task = selected_task

            if st.session_state.get("show_sim", False) and st.session_state.get("sim_vid") == selected_v_id and st.session_state.get("sim_csv") == str(csv_path) and st.session_state.get("sim_task") == selected_task:
                
                @st.cache_data(show_spinner=False)
                def load_and_preprocess_sample(csv_path_str, v_id, task_name):
                    import pandas as pd
                    import yaml
                    from data.preprocessing import (
                        merge_velocity_and_acceleration,
                        clean_labels,
                        engineer_features,
                        derive_sharpness_class
                    )
                    
                    task_path = APP_DIR / f"configs/task/{task_name}.yaml"
                    excluded = []
                    target_col = "Label"
                    if task_path.exists():
                        with open(task_path, 'r') as f:
                            cfg = yaml.safe_load(f)
                            excluded = cfg.get("excluded_labels", [])
                            target_col = cfg.get("target_col", "Label")

                    chunks = []
                    for c in pd.read_csv(csv_path_str, chunksize=50000):
                        chunks.append(c[c['video_id'] == v_id])
                    raw = pd.concat(chunks, ignore_index=True)
                    if raw.empty:
                        return None, None, None, None, None, 0, target_col, excluded
                    
                    merged, base_cols = merge_velocity_and_acceleration(raw)
                    cleaned = clean_labels(merged)
                    
                    pre_clean_len = len(cleaned)
                    if excluded:
                        cleaned = cleaned[~cleaned["Label"].isin(excluded)].copy()
                    dropped_by_task = pre_clean_len - len(cleaned)
                    
                    if target_col == "sharpness_class" and "sharpness_class" not in cleaned.columns:
                        cleaned = derive_sharpness_class(cleaned)

                    eng, f_cols = engineer_features(cleaned.copy(), base_cols)
                    
                    return raw, merged, cleaned, eng, f_cols, dropped_by_task, target_col, excluded
                    
                with st.spinner(f"Processing '{selected_v_id}' under '{selected_task}' rules (cached)..."):
                    raw_df, merged_df, cleaned_df, eng_df, feat_cols, dropped_by_task, task_target_col, excluded_labels = load_and_preprocess_sample(str(csv_path), selected_v_id, selected_task)
                
                if raw_df is None or raw_df.empty:
                    st.warning("No data found for this video ID.")
                else:
                    from data.preprocessing import create_windows
                    import numpy as np
                    
                    st.markdown("---")
                    st.markdown("#### Stage 1: Raw DataFrame Loading")
                    st.markdown(f"Successfully filtered and pinned **{len(raw_df):,}** contiguous rows strictly belonging to `{selected_v_id}`.")
                    
                    s1c1, s1c2, s1c3 = st.columns(3)
                    s1c1.metric("Total Raw Rows", f"{len(raw_df):,}")
                    sensor_cols = [c for c in raw_df.columns if c not in ["video_id", "Frame", "sensor_type", "Label", "activity_type", "sharpness_class"]]
                    s1c2.metric("Unique Sensors Tracked", len(sensor_cols))
                    
                    if "sensor_type" in raw_df.columns:
                        s1c3.metric("Sensor Streams Tagged", raw_df["sensor_type"].nunique())
                        st.caption("Distribution of staggered Sensor streams before horizontal alignment:")
                        st.bar_chart(raw_df["sensor_type"].value_counts())
                    else:
                        s1c3.metric("Sensor Streams Tagged", "Unknown")
                    
                    st.markdown("---")
                    st.markdown("#### Stage 2: Sensor Alignment (`merge_velocity_and_acceleration`)")
                    st.markdown("Physical rows representing the exact same moment in time (Frame) are horizontally zipped together to eliminate vertical staggering.")
                    s2c1, s2c2 = st.columns(2)
                    s2c1.metric("Row Count Transformation", f"{len(raw_df):,} ➔ {len(merged_df):,}", f"Reduced by {len(raw_df)-len(merged_df):,} rows", delta_color="normal")
                    s2c2.metric("Column Count Expansion", f"{len(raw_df.columns)} ➔ {len(merged_df.columns)}", f"Expanded by {len(merged_df.columns)-len(raw_df.columns)} cols", delta_color="normal")
                    
                    st.info("💡 **Suffix Injection Example:** `Right Hand x` splits perfectly into `Right Hand x_vel` and `Right Hand x_acc`.")

                    st.markdown("---")
                    st.markdown("#### Stage 3: Label Cleaning (`clean_labels`)")
                    st.markdown("Dirty string representations are mapped strictly into Neural Network compatible integers (`int64`).")
                    
                    if "Label" in merged_df.columns and "Label" in cleaned_df.columns:
                        original_labels = merged_df["Label"].dropna().unique().tolist()
                        mapping_pairs = []
                        for dirty_lbl in original_labels:
                            sample_frame = merged_df[merged_df["Label"] == dirty_lbl]["Frame"].iloc[0]
                            clean_val_series = cleaned_df[cleaned_df["Frame"] == sample_frame]["Label"]
                            if not clean_val_series.empty:
                                mapping_pairs.append({"Original Discovered String": str(dirty_lbl), "Cleaned Integer Transformation": clean_val_series.iloc[0]})
                        st.table(pd.DataFrame(mapping_pairs))
                        
                        corrupt_drops = len(merged_df) - len(cleaned_df)
                        if corrupt_drops > 0:
                            st.warning(f"⚠️ Dropped {corrupt_drops} corrupted rows where Label string could not be parsed.")
                        else:
                            st.success("✅ No corrupted NaN label rows detected.")
                    else:
                        st.write("Target Column Data Type:", f"`{cleaned_df['Label'].dtype}`")
                        
                    st.markdown("---")
                    st.markdown(f"#### Stage 3.5: Task-Specific Rules (`{selected_task}`)")
                    st.markdown(f"Pulling runtime configuration directly from `backend/configs/task/{selected_task}.yaml`.")
                    
                    tcol1, tcol2 = st.columns(2)
                    tcol1.metric("Target Prediction Column", f"`{task_target_col}`")
                    
                    if excluded_labels:
                        tcol2.metric("Rows Excluded by Task", f"{dropped_by_task:,}", "Dropped", delta_color="inverse")
                        st.info(f"**Task Constraint:** The labels `{excluded_labels}` are strictly prohibited by this task's YAML and were stripped out.")
                    else:
                        tcol2.metric("Rows Excluded by Task", "0")
                        st.success("**Task Constraint:** This task allows all labels. No data was dropped.")

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
                        
                    st.markdown("**Interactive Feature Plotter**")
                    # Isolate genuine engineered columns while masking out administrative/raw data
                    raw_eng = [c for c in eng_df.columns if c not in ["video_id", "Frame", "Label", "activity_type", "sharpness_class", "person_id", "knife_sharpness_score", "_merge"]]
                    pure_eng = [c for c in raw_eng if not c.endswith('_vel') and not c.endswith('_acc')]
                    
                    plot_options = {}
                    for col in pure_eng:
                        # Group kinematic magnitude and 2D planar pairs
                        if col.endswith("_vel_mag") or col.endswith("_acc_mag"):
                            base = col.replace("_vel_mag", " Magnitude").replace("_acc_mag", " Magnitude")
                            if base not in plot_options: plot_options[base] = []
                            plot_options[base].append(col)
                        elif col.endswith("_vel_xy") or col.endswith("_acc_xy"):
                            base = col.replace("_vel_xy", " XY Plane").replace("_acc_xy", " XY Plane")
                            if base not in plot_options: plot_options[base] = []
                            plot_options[base].append(col)
                        elif col.endswith("_vel_xz") or col.endswith("_acc_xz"):
                            base = col.replace("_vel_xz", " XZ Plane").replace("_acc_xz", " XZ Plane")
                            if base not in plot_options: plot_options[base] = []
                            plot_options[base].append(col)
                        elif col.endswith("_vel_yz") or col.endswith("_acc_yz"):
                            base = col.replace("_vel_yz", " YZ Plane").replace("_acc_yz", " YZ Plane")
                            if base not in plot_options: plot_options[base] = []
                            plot_options[base].append(col)
                        else:
                            plot_options[col] = [col]
                            
                    if plot_options:
                        selected_feature = st.selectbox("Select engineered sequence feature to visualize:", list(plot_options.keys()))
                        cols_to_plot = plot_options[selected_feature]
                        st.line_chart(eng_df.set_index("Frame")[cols_to_plot])
                    else:
                        st.warning("No purely engineered columns detected.")

                    st.markdown("---")
                    st.markdown("#### Stage 5: Target Windowing (`create_windows`)")
                        
                    st.markdown(f"Constructing sliding time-series windows (hardcoded `window_size=60`, `stride=30`, `frame_skip_step=2`). Sequence terminates inherently if **`{task_target_col}`** changes or Frame jumps > 1.")
                    
                    try:
                        # Use default parameters as requested
                        X_windows, y_windows, window_meta_df = create_windows(
                            eng_df, 
                            feat_cols, 
                            window_size=60, 
                            stride=30, 
                            target_col=task_target_col, 
                            frame_skip_step=2
                        )
                        
                        if len(X_windows) > 0:
                            sample_shape = X_windows[0].shape
                            st.success(f"Successfully generated **{len(X_windows)} sequences**. Output Tensor Shape: `[{len(X_windows)}, {sample_shape[0]}, {sample_shape[1]}]`")
                            
                            st.markdown(f"**Generated `{task_target_col}` Distribution (Post-Windowing)**")
                            dist_series = pd.Series(y_windows).value_counts().sort_index()
                            # Convert series index to string names for better readability on chart
                            dist_series.index = dist_series.index.map(lambda x: f"{task_target_col}: {x}")
                            st.bar_chart(dist_series)
                        else:
                            st.warning("Not enough continuous frames to build a 60-frame window from this subset.")
                    except Exception as e:
                        st.error(f"Windowing failed: {e}")
