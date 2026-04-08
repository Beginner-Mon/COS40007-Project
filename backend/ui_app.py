import streamlit as st

st.set_page_config(page_title="Automated ML Pipeline", layout="wide")

st.title("🚀 Pipeline Manager")

pg1 = st.Page("ui/pipeline_architecture.py", title="Pipeline Architecture & Data Flow", icon="⚙️")
pg2 = st.Page("ui/data_preprocessing.py", title="Data Preprocessing & EDA", icon="📊")
pg3 = st.Page("ui/model_training.py", title="Model Training", icon="🔥")
pg4 = st.Page("ui/model_inference.py", title="Model Inference", icon="🎯")

pg = st.navigation([pg1, pg2, pg3, pg4])
pg.run()
