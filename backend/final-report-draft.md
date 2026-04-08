# Final Report Draft (Project-Specific)

This draft is based on the current backend implementation, configs, and tracked training runs.

## A. Title Page

### Suggested content
Studio Session No.: [fill]
Group No.: [fill]
Theme No.: [fill]
Group Members:
- [Name] ([Student Number])
- [Name] ([Student Number])
- [Name] ([Student Number])

### Figure(s) to insert
- No figure needed.

---

## 1. Introduction

### 1.1 Background and Motivation

### Suggested content
The project addresses motion-based activity recognition in meat processing tasks using human movement time-series data. The practical motivation is to support safer and more consistent processing operations by automatically recognizing worker actions such as boning and slicing, and by studying whether motion patterns can be associated with knife sharpness levels.

From an engineering perspective, the project combines sensor data processing, sequence modeling, and deployment-oriented tooling. The expected users are researchers, process engineers, and training supervisors who want to (1) classify activities from movement signals, (2) inspect the quality of temporal data, and (3) evaluate model behavior across participants and task conditions.

This project was selected because it offers a realistic end-to-end AI workflow: raw industrial-style data ingestion, noisy label handling, time-window construction, deep learning model training, and demonstrator development.

### Figure(s) to insert
- Figure 1: Problem context diagram (motion capture input -> AI model -> predicted activity/sharpness -> engineering decision support).
- Figure 2: Stakeholder and use-case map (researcher, trainer, process engineer).

---

### 1.2 Project Objectives

### Suggested content
The project objectives are:

1. To build an AI pipeline that classifies boning vs slicing actions from motion features.
2. To investigate a 3-class knife sharpness classification task (blunt, medium, sharp) derived from sharpness scores.
3. To develop a full multi-class activity recognition model using label-cleaned temporal windows.
4. To compare sequence modeling approaches available in the codebase (BiLSTM, GRU, TCN) and justify the final model used.
5. To implement a reproducible training/evaluation workflow with Hydra and MLflow.
6. To deliver a demonstrator that supports dataset upload, preprocessing inspection, EDA, and model training logs.

Primary questions addressed:

1. Can temporal windows of engineered velocity/acceleration features reliably separate target classes?
2. How much do grouped split strategies by video_id reduce leakage risk compared to naive random frame splits?
3. Which task is easier/harder under the current feature pipeline, measured by validation loss and validation accuracy?

Benefits to users:

1. Faster and more consistent activity labeling support.
2. Better interpretability of movement behavior through preprocessing and EDA views.
3. A reusable template for sensor-based sequence modeling in engineering contexts.

### Figure(s) to insert
- Figure 3: Objectives-to-method mapping table (objective, method, metric).

---

### 1.3 Summary of Outcomes

### Suggested content
The implemented pipeline processed 26 videos (P1/P2 combined) and generated task-specific training windows from cleaned and merged sensor streams. The data preprocessing stage corrected inconsistent labels and applied automatic boning label correction (Label 5 -> 8) where needed.

Key quantitative outcomes from tracked runs:

1. Boning vs Slicing (k-fold): best recorded average validation loss = 0.00963, with best-fold validation accuracy up to 0.9868.
2. Knife Sharpness (k-fold): average validation loss = 0.2135, with best-fold validation accuracy around 0.4983.
3. Activity Recognition (holdout): validation loss range = 0.1451 to 0.1873, validation accuracy range = 0.6622 to 0.6964 across recorded runs.

These results suggest that binary activity discrimination is the strongest task under the current pipeline, while sharpness classification remains more challenging.

### Figure(s) to insert
- Figure 4: Summary results bar chart (task vs best validation accuracy).
- Figure 5: Validation loss comparison chart (task vs loss).

---

## 2. Dataset

### 2.1 Data Source

### Suggested content
The dataset is organized by participant and activity:

- dataset/P1/Boning (11 files)
- dataset/P1/Slicing (3 files)
- dataset/P2/Boning (9 files)
- dataset/P2/Slicing (3 files)

Total source files: 26 Excel files.

Each Excel file includes at least three sheets used by the pipeline:

1. Segment Velocity
2. Segment Acceleration
3. Markers

The loader extracts motion features (3D coordinates for key body segments), frame indices, and labels, then injects metadata including person_id, activity_type, knife_sharpness_score, sensor_type, and video_id. Labels are derived from explicit label/marker columns and frame ranges in the Markers sheet.

### Figure(s) to insert
- Figure 6: Dataset folder tree and file count summary.
- Figure 7: Sample Excel schema (sheets and key columns).
- Figure 8: Example metadata extraction from filename (video_id and sharpness score parsing).

---

### 2.2 Data Processing

### Suggested content
The preprocessing pipeline follows these stages:

1. Read all CSV exports from output_data.
2. Merge Segment Velocity and Segment Acceleration rows by (video_id, Frame) into a single aligned frame record.
3. Normalize and clean labels using a mapping dictionary for inconsistent text formats.
4. Apply activity-aware correction for boning segments where label 5 is remapped to 8.
5. Build engineered motion features from merged velocity and acceleration channels.
6. Segment continuous runs and generate windows with task-specific rules.
7. Pad/truncate windows to fixed length (60 timesteps).
8. Replace NaN/Inf values using numeric sanitation.

Observed data scale in the current project snapshot:

- Raw rows across all CSV exports in output_data (including Segment Position exports): 1,590,879
- Raw rows actually used by train.py (P1/P2 boning+slicing CSV files): 1,060,586
- Unique video_id values: 26
- Rows after velocity-acceleration merge: 530,293
- Engineered features per timestep: 145

Windowed dataset size by task:

1. Activity Recognition: 9,369 windows (9 classes)
2. Boning vs Slicing: 6,202 windows (2 classes)
3. Knife Sharpness: 6,202 windows (3 classes)

### Figure(s) to insert
- Figure 9: End-to-end preprocessing flowchart (raw Excel -> cleaned windows tensor).
- Figure 10: Class distribution bar charts for each task (window-level counts).
- Figure 11: Before/after label-cleaning example table.
- Figure 12: Row-count reduction chart (raw rows -> merged rows -> windowed samples).

---

## 3. AI Model Development

### 3.1 Feature Engineering / Feature Extraction

### Suggested content
Feature engineering is performed after sensor merging. The pipeline computes motion-intensity and energy-based features from 3D velocity and acceleration channels.

Feature groups include:

1. Per-segment velocity magnitude features (22):

$$
|v| = \sqrt{v_x^2 + v_y^2 + v_z^2}
$$

2. Per-segment acceleration magnitude features (22):

$$
|a| = \sqrt{a_x^2 + a_y^2 + a_z^2}
$$

3. Total body energy (1):

$$
E_{total} = \sum_{joints}(v_x^2+v_y^2+v_z^2+a_x^2+a_y^2+a_z^2)
$$

4. Pairwise planar magnitudes for arm/leg joints (96 total for velocity+acceleration across xy/xz/yz planes).
5. Energy ratios (4): upper/lower and left/right energy distribution.

Total engineered feature count: 145.

For normalization, StandardScaler is fitted on training windows only and then applied to validation windows to avoid data leakage.

### Figure(s) to insert
- Figure 13: Feature taxonomy diagram (44 magnitudes + 96 pairwise + 1 total energy + 4 ratios).
- Figure 14: Formula panel for main engineered features.
- Figure 15: Example feature trajectory over frames for one video.

---

### 3.2 Train/Test Split

### Suggested content
The final implementation uses notebook-specific split and window policies rather than one global train/test setting. The comparison below is extracted from the three task notebooks and normalized by alias:

- window_size <-> WINDOW_SIZE
- stride <-> WINDOW_STRIDE
- frame_skip_step <-> FRAME_SKIP_STEP

#### Cross-notebook comparison (selected groups)

##### A) Windowing and frame sampling

| Setting | activity_recognition.ipynb | boning_vs_slicing.ipynb | knife_sharpness.ipynb |
|---|---|---|---|
| Window size | 60 | 60 | 60 |
| Window stride | 30 (default; label overrides exist) | 30 | 30 |
| Frame skip step | Default 2, label-specific overrides | 2 | 1 |
| Label-specific window rules | Yes (`LABEL_SPECIFIC_RULES`) | No | No |
| Source anchors | `#VSC-c77af539` (lines 626-789) | `#VSC-9af3c088` (lines 723-807) | `#VSC-068a8913` (lines 915-996) |

Activity recognition label-specific overrides:

- Label 4: window_size=60, window_stride=30, frame_skip_step=3
- Label 0: window_size=60, window_stride=30, frame_skip_step=2
- Label 6: window_size=60, window_stride=15, frame_skip_step=1
- Label 7: window_size=60, window_stride=20, frame_skip_step=1
- Label 5: window_size=60, window_stride=20, frame_skip_step=1
- Label 8: window_size=60, window_stride=20, frame_skip_step=2

##### B) Data split strategy

| Setting | activity_recognition.ipynb | boning_vs_slicing.ipynb | knife_sharpness.ipynb |
|---|---|---|---|
| Primary split method | Holdout by explicit `TEST_VIDEO_IDS` | StratifiedGroupKFold | StratifiedGroupKFold |
| Group key | `video_id` | `video_id` | `video_id` |
| Key split parameters | `val_split=0.2` + manual holdout IDs | `n_splits=10`, `shuffle=True`, `random_state=42` | `n_splits=10`, `shuffle=True`, `random_state=42` |
| Holdout IDs used | `MVN-J-Boning-90-004`, `MVN-S-Slicing-63-001`, `MVN-S-Boning-89-004` | Not used | Not used |
| Source anchors | `#VSC-ebc27c58` (lines 420-492), `#VSC-2fbbfe05` (lines 47-76) | `#VSC-8789fca0` (lines 846-880) | `#VSC-38a62d39` (lines 1035-1069) |

The grouped strategy remains critical for k-fold tasks because frame-level samples from the same video are temporally correlated; grouping by `video_id` reduces leakage and improves realism of validation estimates.

##### C) Model hyperparameters

| Hyperparameter | activity_recognition.ipynb | boning_vs_slicing.ipynb | knife_sharpness.ipynb |
|---|---|---|---|
| hidden_size | 128 | 128 | 64 |
| num_layers | 1 | 1 | 1 |
| dropout | 0.4 | 0.4 | 0.2 |
| Source anchors | `#VSC-71c0b297` (lines 1056-1078) | `#VSC-8581705e` (lines 1100-1114) | `#VSC-3809bb13` (lines 1264-1279) |

##### D) Feature-engineering toggles

| Setting | activity_recognition.ipynb | boning_vs_slicing.ipynb | knife_sharpness.ipynb |
|---|---|---|---|
| Toggle mechanism | No explicit toggle dict | No explicit toggle dict | Explicit `ENABLE_FEATURE_GROUPS` dict |
| Core engineered set | velocity magnitude + acceleration magnitude + total_body_energy (45 features) | Adds axis-pair magnitudes + energy distribution ratios | Selective inclusion by toggle flags |
| axis_pair | Always excluded (not defined in training feature set) | Always included | `False` |
| jerk_dynamics | Not defined | Not defined | `True` |
| abruptness_ratios | Not defined | Not defined | `True` |
| asymmetry_compensation | Not defined | Not defined | `True` |
| coordination | Not defined | Not defined | `True` |
| Source anchors | `#VSC-6d40d8c3` (lines 337-388) | `#VSC-b1e0629d` (lines 342-491) | `#VSC-6560a3df` (feature section with `ENABLE_FEATURE_GROUPS`) |

### Figure(s) to insert
- Figure 16: Split strategy diagram (video-level grouped split).
- Figure 17: Holdout example showing train vs validation video_id allocation.

---

### 3.3 Training Model

### Suggested content
Training is implemented in PyTorch with Hydra/MLflow-compatible tracking patterns, while notebook experiments apply task-specific model capacity and windowing behavior (summarized in Section 3.2).

Across the three notebooks:

1. Activity Recognition and Boning vs Slicing use the same recurrent capacity (`hidden_size=128`, `num_layers=1`, `dropout=0.4`).
2. Knife Sharpness uses a smaller recurrent capacity (`hidden_size=64`, `num_layers=1`, `dropout=0.2`).
3. Window shape is fixed at 60 timesteps for all tasks, but frame sampling differs (`FRAME_SKIP_STEP=2` for Boning vs Slicing, `FRAME_SKIP_STEP=1` for Knife Sharpness, and label-specific rules for Activity Recognition).

This indicates that the codebase keeps a shared sequence-model template while tuning sampling and model capacity per task difficulty and label structure.

Model options in the codebase:

1. BiLSTM (bidirectional recurrent encoder + fully connected head)
2. GRU
3. TCN (temporal convolution network)

Training stabilization methods:

1. Early stopping
2. Checkpoint saving (best.pt)
3. ReduceLROnPlateau scheduler

All runs log parameters and fold metrics to MLflow for traceability.

### Figure(s) to insert
- Figure 18: Final selected architecture diagram (BiLSTM).
- Figure 19: Training loop and callback sequence chart.
- Figure 20: MLflow screenshot of run metrics/parameters.

---

### 3.4 Evaluation of AI Model

### Suggested content
Evaluation uses validation loss and validation accuracy from MLflow metrics. For k-fold experiments, both average validation loss across folds and best-fold metrics are reported.

Suggested result table content:

| Task | Split | Metric Summary | Best Observed Result |
|---|---|---|---|
| Boning vs Slicing | 10-fold grouped CV | avg_val_loss = 0.00963 | best fold: val_loss = 0.00161, val_acc = 0.9868 |
| Knife Sharpness | 10-fold grouped CV | avg_val_loss = 0.21347 | best fold: val_loss = 0.11499, val_acc = 0.4983 |
| Activity Recognition | Holdout by video_id | val_loss range = 0.1451 to 0.1873 | val_acc range = 0.6622 to 0.6964 |

Interpretation:

1. Binary activity classification is highly separable under current feature engineering.
2. Full multi-class activity recognition is moderate and likely limited by class overlap and label complexity.
3. Sharpness classification is the most difficult task and may require improved labels/features or additional context.

Note for report transparency:

- models/eval_results_P2.yaml records loss = 0.2163 and accuracy = 29.7346. Since accuracy should typically be in [0, 1] for this implementation, re-validation of this output is recommended before using it as a final headline metric.

### Figure(s) to insert
- Figure 21: Confusion matrix for each task (especially boning_vs_slicing and activity_recognition).
- Figure 22: Per-task metric comparison bar chart.
- Figure 23: Fold-wise validation loss boxplot (k-fold tasks).

---

## 4. AI Demonstrator

### Suggested content
A Streamlit-based demonstrator (ui_app.py) is implemented with three interactive pages:

1. Pipeline Architecture & Data Flow
2. Data Preprocessing & EDA
3. Model Training

Demonstrator capabilities:

1. Upload and validate custom dataset ZIPs with required P1/P2 and Boning/Slicing structure.
2. Run preprocessing and visualize intermediate transformations.
3. Perform EDA (histograms, correlation heatmaps, frame-wise time series).
4. Launch training runs with configurable task, split strategy, model architecture, and hyperparameters.
5. Track training via live logs and MLflow links.

Required inputs:

- Existing output_data CSVs, or uploaded dataset ZIP.
- User-selected task/hyperparameters from sidebar controls.

Produced outputs:

- Interactive plots and dataset diagnostics.
- Training progress logs and fold/epoch status.
- MLflow run links for experiment analysis.

### Figure(s) to insert
- Figure 24: Screenshot of Pipeline Architecture page.
- Figure 25: Screenshot of EDA page with feature plots.
- Figure 26: Screenshot of Model Training page with MLflow status.
- Figure 27: Screenshot of live training logs/progress bar.

---

## 5. Conclusions

### Suggested content
This project delivered an end-to-end AI workflow for motion-based recognition tasks, from raw Excel ingestion to trained sequence models and an interactive demonstrator.

Final outcomes:

1. A robust preprocessing pipeline that merges sensor modalities, cleans labels, and generates fixed-length windows.
2. Strong performance for boning vs slicing classification.
3. Moderate performance for full activity recognition and weaker sharpness classification, indicating meaningful room for further improvement.

Main technical challenges and resolutions:

1. Inconsistent labels across files: solved using centralized label mapping and activity-aware correction rules.
2. Temporal leakage risk: addressed through grouped split strategies by video_id.
3. Class imbalance and hard examples: addressed using Focal Loss and training callbacks.
4. Reproducibility and experiment tracking: handled via Hydra config snapshots and MLflow logging.

Key learnings:

1. Data preprocessing and split design strongly influence sequence model performance.
2. Engineered kinematic features can provide high separability for simpler activity tasks.
3. More complex targets (e.g., sharpness class) require richer signal context, improved annotation quality, or model redesign.

### Figure(s) to insert
- Figure 28: Lessons learned summary diagram (data, model, evaluation, deployment).

---

## 6. References

### Suggested references (edit to your final citation style)

1. Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.
2. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv:1406.1078.
3. Bai, S., Kolter, J. Z., and Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271.
4. Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV 2017.
5. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.
6. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830.
7. Hydra Documentation. https://hydra.cc/docs/intro/
8. MLflow Documentation. https://mlflow.org/docs/latest/index.html
9. Streamlit Documentation. https://docs.streamlit.io/
10. (If applicable) Motion capture system/vendor documentation used with this dataset.
11. ChatGPT prompt logs used for coding/reporting assistance (include date and prompt text as required by subject policy).

### Figure(s) to insert
- No figure needed.

---

## 7. Appendix A

### Suggested content
Include:

1. Source code/notebooks link:
- GitHub repository: [fill]
- Notebook folder snapshot: backend/notebook/

2. Intermediate data and final model links:
- output_data CSV exports
- models/best.pt and evaluation YAML
- MLflow experiment artifact folder (if required)

3. Run instructions (example):

```bash
# Generate CSV from dataset Excel files
python -m data.loader --output-dir output_data

# Train (example task)
python train.py task=boning_vs_slicing

# Evaluate saved model
python eval.py model_path=models/best.pt data.participant=P2

# Launch demonstrator
streamlit run ui_app.py
```

4. Public demo link (if hosted): [fill]

### Figure(s) to insert
- Figure A1: Repository structure screenshot.
- Figure A2: Quick-start workflow screenshot.

---

## Information Needed From You To Finalize

Please provide these details so I can produce a final submission-ready version:

1. Title page details (studio/group/theme/member names and IDs).
2. Your actual motivation and intended user scenario (1-2 paragraphs in your own words).
3. Which runs you want to present as final (run IDs or timestamps), especially for activity_recognition and knife_sharpness.
4. Whether to include GRU/TCN comparison results (if you have them) or keep BiLSTM as the main model.
5. Screenshots you want to include from the demonstrator pages.
6. Required citation style (IEEE, APA, Harvard, etc.).
