import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from ui.utils import APP_DIR

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
