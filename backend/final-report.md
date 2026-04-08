# Final Report — Motion-Based Activity Recognition and Knife Sharpness Classification

> **Studio Session No.:** [Fill]  
> **Group No.:** [Fill]  
> **Theme No.:** [Fill]  
>
> **Group Members:**
> - [Name] ([Student Number])
> - [Name] ([Student Number])
> - [Name] ([Student Number])

---

## 1. Introduction

### 1.1 Background and Motivation

The meat processing industry demands both speed and safety on the production floor. Workers perform repetitive manual actions — boning, slicing, reaching, pulling, and placing — under time pressure, and the sharpness of their knives directly affects cutting efficiency, product quality, and risk of injury. Despite these stakes, activity monitoring and equipment assessment in such environments remain largely manual and subjective.

This project applies AI to motion-capture time-series data recorded from meat processing workers. The core idea is that 3D motion patterns — captured as segment velocity and acceleration signals across the full body — contain sufficient information to (1) automatically classify what the worker is doing at any moment, and (2) infer the sharpness condition of the knife being used. By making these judgements automatic, the system offers value to several user groups:

- **Process engineers** seeking to quantify task duration, worker consistency, and line throughput without manual observation.
- **Training supervisors** who need objective feedback on trainee technique and movement quality.
- **Safety and quality researchers** interested in the relationship between knife sharpness, motion intensity, and ergonomic risk.

This project was selected because it represents a realistic end-to-end AI engineering workflow: raw industrial-grade sensor data ingestion, noisy label handling, temporal windowing, deep learning model training, experiment tracking, and interactive demonstrator development. These challenges mirror the kinds of problems encountered in real-world deployable AI systems and provide a strong learning foundation in applied machine learning.

### 1.2 Project Objectives

The project objectives are:

1. **Binary activity classification** — Build an AI pipeline that classifies boning versus slicing actions from engineered motion features.
2. **Knife sharpness classification** — Investigate a 3-class sharpness task (blunt, medium, sharp) derived from sharpness scores embedded in the dataset filenames.
3. **Full multi-class activity recognition** — Develop a 9-class model recognising all labelled activities (Idle, Walking, Steeling, Reaching, Cutting, Slicing, Pulling, Placing/Manipulating, Dropping) from cleaned temporal windows.
4. **Model architecture comparison** — Compare sequence modelling approaches available in the codebase (BiLSTM, GRU, TCN) and justify the final selection for each task.
5. **Reproducible workflow** — Implement a fully reproducible training and evaluation pipeline using Hydra for configuration management and MLflow for experiment tracking.
6. **Interactive demonstrator** — Deliver a Streamlit-based demonstrator supporting dataset upload, preprocessing inspection, exploratory data analysis (EDA), and live model training with progress monitoring.

**Primary research questions:**

1. Can temporal windows of engineered velocity and acceleration features reliably separate target classes across all three tasks?
2. How much do grouped split strategies (by `video_id`) reduce data leakage compared to naive random frame-level splits?
3. Which task is easiest/hardest under the current feature pipeline, as measured by validation loss and validation accuracy?

**Benefits to users:**

- Faster and more consistent activity labelling support for process documentation.
- Better interpretability of worker movement behaviour through preprocessing visualisations and EDA views.
- A reusable, modular template for sensor-based sequence modelling in engineering and industrial contexts.

### 1.3 Summary of Outcomes

The implemented pipeline processed 26 Excel files (P1 and P2 combined — 11 boning + 3 slicing for P1, 9 boning + 3 slicing for P2) and generated task-specific training windows from cleaned and merged sensor streams. The data preprocessing stage corrected inconsistent text labels across files and applied automatic boning label correction (Label 5 → 8) where needed.

Key quantitative outcomes from tracked MLflow runs:

| Task | Split Strategy | Key Metric | Best Observed |
|---|---|---|---|
| Boning vs Slicing | 10-fold StratifiedGroupKFold | avg\_val\_loss = **0.00963** | best fold: val\_loss = 0.00161, val\_acc = **0.9868** |
| Knife Sharpness | 10-fold StratifiedGroupKFold | avg\_val\_loss = **0.2135** | best fold: val\_loss = 0.1150, val\_acc = **0.4983** |
| Activity Recognition | Holdout by video\_id | val\_loss range = 0.1451 – 0.1873 | val\_acc range = **0.6622 – 0.6964** |

These results demonstrate that binary activity discrimination (boning vs slicing) is highly separable under the current feature engineering pipeline, full multi-class activity recognition achieves moderate accuracy, and sharpness classification remains the most challenging task — likely requiring richer signal context, improved annotation quality, or model redesign.

---

## 2. Dataset

### 2.1 Data Source

The dataset is organised by participant and activity type in the following folder structure:

```
dataset/
├── P1/
│   ├── Boning/    (11 Excel files)
│   └── Slicing/   (3 Excel files)
└── P2/
    ├── Boning/    (9 Excel files)
    └── Slicing/   (3 Excel files)
```

**Total source files:** 26 Excel workbooks (`.xlsx`)

Each Excel file contains at least three sheets used by the pipeline:

1. **Segment Velocity** — 3D velocity vectors (x, y, z) for 22 body segments per frame.
2. **Segment Acceleration** — 3D acceleration vectors (x, y, z) for the same 22 body segments per frame.
3. **Markers** — Frame ranges and associated activity labels, used to assign ground-truth labels to each frame.

The 22 tracked body segments are:

| Region | Segments |
|---|---|
| Spine and Head | L5, L3, T12, T8, Neck, Head |
| Right Arm | Right Shoulder, Right Upper Arm, Right Forearm, Right Hand |
| Left Arm | Left Shoulder, Left Upper Arm, Left Forearm, Left Hand |
| Right Leg | Right Upper Leg, Right Lower Leg, Right Foot, Right Toe |
| Left Leg | Left Upper Leg, Left Lower Leg, Left Foot, Left Toe |

The data loader (`data/base.py → ExcelReader`) reads each sheet and injects the following metadata columns, parsed from the file path and filename:

- `person_id` — Extracted from the directory path (P1 or P2).
- `activity_type` — Extracted from the directory path (boning or slicing).
- `knife_sharpness_score` — Parsed from the filename using a regex pattern `-(\\d{2,3})-` (e.g., `MVN-J-Boning-90-004.xlsx` → sharpness score = 90).
- `sensor_type` — The sheet name (e.g., "Segment Velocity").
- `video_id` — The file stem (e.g., `MVN-J-Boning-90-004`).

Labels are assigned to each frame by reading the Markers sheet and mapping frame ranges to activity strings. Where existing Label or Marker columns contain NaN gaps, the loader backfills labels from the marker frame ranges.

### 2.2 Data Processing

The preprocessing pipeline follows a strict sequence of stages, implemented across `data/loader.py`, `data/base.py`, and `data/preprocessing.py`:

**Stage 1 — CSV Loading**

The loader (`data/loader.py → DataLoader`) reads all Excel files per participant per activity, applies label and feature schema validation, and exports the processed DataFrames as CSV files to `output_data/` (e.g., `P1_boning.csv`, `P2_slicing.csv`). The training script (`train.py`) then loads these four CSV files and concatenates them into a single raw DataFrame.

**Stage 2 — Sensor Merging**

The raw DataFrame contains interleaved rows: for each `(video_id, Frame)` pair there are separate rows for Segment Velocity and Segment Acceleration. The `merge_velocity_and_acceleration()` function performs an inner join on `(video_id, Frame)` to horizontally align these sensor streams. This halves the row count and doubles the feature columns:

- Each base feature column (e.g., `Right Hand x`) splits into `Right Hand x_vel` and `Right Hand x_acc`.
- The merge uses `validate="one_to_one"` to ensure frame-level integrity.

**Stage 3 — Label Cleaning**

The `clean_labels()` function addresses inconsistent label strings across files. A centralised `LABEL_MAPPING` dictionary maps 15+ variant strings to consistent integer codes:

| Original String Variants | Mapped Integer |
|---|---|
| `'0- Idle'`, `'0 - Idle'` | 0 |
| `'1- Walking'`, `'1 - Walking'` | 1 |
| `'2- Steeling'`, `'2 - Steeling'` | 2 |
| `'3- Reaching'`, `'3 - Reaching'` | 3 |
| `'4- Cutting'`, `'4 - Cutting'`, `'4 - Cutting (Big Pieces)'`, etc. | 4 |
| `'5 - Slicing'` | 5 |
| `'6 - Pulling'` | 6 |
| `'7 - Placing/ Manipulating'` | 7 |
| `'5- Dropping'`, `'5 - Dropping'`, `'8 - Dropping'` | 8 |

A critical activity-aware override is also applied: for boning activity files, any row with Label 5 ("Dropping" mislabelled as "Slicing") is automatically corrected to Label 8. This prevents label contamination between the two activity types.

**Stage 4 — Task-Specific Filtering**

Each task configuration (defined in YAML under `configs/task/`) specifies which labels to exclude and which target column to use:

| Task | Target Column | Excluded Labels | Effective Classes |
|---|---|---|---|
| Boning vs Slicing | `activity_type` | [0, 1, 2, 3] | 2 (boning, slicing) |
| Knife Sharpness | `sharpness_class` | [0, 1, 2, 3] | 3 (blunt, medium, sharp) |
| Activity Recognition | `Label` | [] | 9 (labels 0–8) |

For the knife sharpness task, a `derive_sharpness_class()` function bins the continuous `knife_sharpness_score` into three categories: **blunt** (< 70), **medium** (70–84), **sharp** (≥ 85).

**Stage 5 — Feature Engineering**

Detailed in Section 3.1.

**Stage 6 — Windowing**

The `create_windows()` function segments the merged, cleaned DataFrame into fixed-length temporal windows. Continuous runs are first identified — consecutive frames sharing the same `video_id` and target label with no frame gaps. Each run is then sampled according to:

- `frame_skip_step` — downsample by taking every N-th frame.
- `window_size` — number of sampled frames per window (default: 60).
- `stride` — step size between consecutive window starts (default: 30).

Windows that are shorter than the target length are kept and zero-padded in Stage 7.

**Stage 7 — Padding**

`pad_windows_to_60()` ensures all windows have exactly 60 timesteps. Short windows are zero-padded; windows exceeding the target are truncated.

**Stage 8 — Numeric Sanitisation**

`clean_features()` replaces any remaining NaN, +Inf, or -Inf values with 0.0 to prevent training instability.

**Observed data scale:**

| Metric | Value |
|---|---|
| Raw rows across all CSV exports | ~1,590,879 |
| Raw rows used by training (P1+P2 boning+slicing) | ~1,060,586 |
| Unique video\_id values | 26 |
| Rows after velocity–acceleration merge | ~530,293 |
| Engineered features per timestep | 145 |
| Windowed samples — Activity Recognition | ~9,369 windows (9 classes) |
| Windowed samples — Boning vs Slicing | ~6,202 windows (2 classes) |
| Windowed samples — Knife Sharpness | ~6,202 windows (3 classes) |

---

## 3. AI Model Development

### 3.1 Feature Engineering

Feature engineering is performed after sensor merging by the `engineer_features()` function in `data/preprocessing.py`. The pipeline computes motion-intensity and energy-based features from the 3D velocity and acceleration channels of all 22 body segments.

**Feature Group 1 — Per-Segment Velocity Magnitude (22 features)**

For each of the 22 body segments, the 3D velocity magnitude is computed:

$$|v| = \sqrt{v_x^2 + v_y^2 + v_z^2}$$

This produces one scalar per segment per frame, capturing the overall speed of that segment regardless of direction (e.g., `Right_Hand_vel_mag`, `Head_vel_mag`).

**Feature Group 2 — Per-Segment Acceleration Magnitude (22 features)**

Similarly, the 3D acceleration magnitude is computed for each segment:

$$|a| = \sqrt{a_x^2 + a_y^2 + a_z^2}$$

This captures the intensity of change in velocity, useful for detecting sudden movements, impacts, or transitions between actions.

**Feature Group 3 — Total Body Energy (1 feature)**

A single global energy metric sums the squared velocity and acceleration components across all segments and all axes:

$$E_{total} = \sum_{joints}(v_x^2 + v_y^2 + v_z^2 + a_x^2 + a_y^2 + a_z^2)$$

This provides a holistic measure of full-body motion intensity per frame.

**Feature Group 4 — Pairwise Planar Magnitudes (96 features)**

For arm and leg joints (16 segments: 4 right arm, 4 left arm, 4 right leg, 4 left leg), the pipeline computes 2D planar magnitudes across three planes — XY, XZ, and YZ — for both velocity and acceleration:

$$M_{xy} = \sqrt{c_x^2 + c_y^2}$$

This yields 16 segments × 3 planes × 2 modalities (velocity + acceleration) = **96 features**. These capture directional motion patterns (e.g., lateral vs vertical movement) that are discriminative for different activities.

**Feature Group 5 — Energy Distribution Ratios (4 features)**

Body-regional energy ratios quantify how motion energy is distributed across body regions:

- **Upper energy ratio:** $\frac{E_{upper}}{E_{upper} + E_{lower} + \epsilon}$
- **Lower energy ratio:** $\frac{E_{lower}}{E_{upper} + E_{lower} + \epsilon}$
- **Left energy ratio:** $\frac{E_{left}}{E_{left} + E_{right} + \epsilon}$
- **Right energy ratio:** $\frac{E_{right}}{E_{left} + E_{right} + \epsilon}$

where $\epsilon = 10^{-8}$ prevents division by zero. Upper body includes spine, head, and arm segments; lower body includes leg segments. These ratios are particularly informative for distinguishing activities that primarily engage upper versus lower body (e.g., cutting vs walking).

**Total engineered feature count: 145** (22 + 22 + 1 + 96 + 4).

**Normalisation:**

Standard normalisation (`StandardScaler` from scikit-learn) is applied after the train/validation split. The scaler is fitted exclusively on training windows and then applied to validation windows, ensuring no data leakage from the validation set into the training statistics.

### 3.2 Train/Test Split

The project uses two validation strategies, selected per task via the Hydra YAML configuration:

#### Strategy 1: StratifiedGroupKFold (K-Fold Cross-Validation)

Used by the **Boning vs Slicing** and **Knife Sharpness** tasks.

- **Implementation:** `sklearn.model_selection.StratifiedGroupKFold` with `n_splits=10`, `shuffle=True`, `random_state=42`.
- **Group key:** `video_id` — all windows from the same video are guaranteed to appear in only one fold (either training or validation), preventing temporal leakage.
- **Stratification:** Ensures each fold preserves the class distribution of the target variable.

This grouped strategy is critical because frame-level samples from the same video are highly temporally correlated. A naive random split would allow frames from the same video to appear in both training and validation sets, inflating performance metrics and producing misleadingly optimistic results.

#### Strategy 2: Holdout (Train/Test Split)

Used by the **Activity Recognition** task.

- **Implementation:** `sklearn.model_selection.GroupShuffleSplit` with `test_size=0.2`.
- **Group key:** `video_id`.
- **Manual override:** The configuration supports specifying explicit `test_video_ids` for deterministic holdout allocation.

When no explicit holdout IDs are set, the holdout module automatically performs a grouped random split, selecting approximately 20% of the video groups for validation while keeping all windows from each video together.

#### Cross-Task Configuration Summary

| Setting | Boning vs Slicing | Knife Sharpness | Activity Recognition |
|---|---|---|---|
| Split strategy | 10-fold grouped CV | 10-fold grouped CV | Holdout by video\_id |
| Group key | `video_id` | `video_id` | `video_id` |
| Frame skip step | 2 | 2 | Label-specific (1–3) |
| Window size | 60 | 60 | 60 |
| Window stride | 30 | 30 | Label-specific (15–30) |

The Activity Recognition task uses **label-specific windowing rules** defined in `configs/task/activity_recognition.yaml`. These per-label overrides adjust stride and frame skip step to handle class imbalance and varying action durations:

| Label | Activity | Window Stride | Frame Skip Step |
|---|---|---|---|
| 0 | Idle | 30 | 2 |
| 4 | Cutting | 30 | 3 |
| 5 | Slicing | 20 | 1 |
| 6 | Pulling | 15 | 1 |
| 7 | Placing/Manipulating | 20 | 1 |
| 8 | Dropping | 20 | 2 |

Under-represented labels use smaller strides (more overlap → more windows) and smaller skip steps (denser sampling → finer temporal resolution).

### 3.3 Training Model

Training is implemented in PyTorch with a modular architecture supporting multiple model types and validation strategies.

#### Model Architectures

Three sequence model architectures are implemented in the `model/` package:

**1. BiLSTM (Bidirectional Long Short-Term Memory)** — `model/bilstm.py`

The primary model used across experiments. Architecture:

```
Input (batch, 60, 145)
    ↓
Bidirectional LSTM (hidden_size × 2 directions)
    ↓
Temporal Mean Pooling (over time dimension)
    ↓
BatchNorm1d → Linear(hidden×2, hidden) → Dropout → ReLU
    ↓
Linear(hidden, num_classes)
    ↓
Output logits
```

Key design decisions:
- Bidirectional processing allows the model to consider both past and future context within each window.
- Mean pooling over time (rather than taking the last hidden state) produces a more robust sequence representation.
- Batch normalisation before the fully connected layer stabilises training.

**2. GRU (Gated Recurrent Unit)** — `model/gru.py`

A lighter alternative to LSTM with fewer parameters:

```
Input (batch, 60, 145)
    ↓
GRU (unidirectional)
    ↓
Last time step output
    ↓
FC(hidden, 128) → Dropout → ReLU
    ↓
FC(128, 128) → Dropout → ReLU
    ↓
FC(128, num_classes)
```

The GRU uses two fully connected blocks instead of one, and takes only the last time step rather than pooling over the full sequence.

**3. TCN (Temporal Convolutional Network)** — `model/tcn.py`

A non-recurrent alternative using dilated causal convolutions:

```
Input (batch, 60, 145) → transpose → (batch, 145, 60)
    ↓
TemporalConvNet (stacked TemporalBlocks with exponentially increasing dilation)
    ↓
Last time step output
    ↓
FC(hidden, 128) → Dropout → ReLU
    ↓
FC(128, 128) → Dropout → ReLU
    ↓
FC(128, num_classes)
```

Each `TemporalBlock` contains two dilated 1D convolutions with residual connections, allowing the network to capture long-range temporal dependencies without recurrence.

#### Training Configuration

Default hyperparameters (from `configs/train.yaml`):

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1×10⁻⁴ |
| Weight decay | 1×10⁻⁵ |
| Gradient clipping | 1.0 |
| Epochs | 30 |
| Batch size | 32 |
| Hidden size | 128 |
| Number of layers | 1 |
| Dropout | 0.4 |

#### Loss Function

The pipeline uses **Focal Loss** (`training/loss.py`) by default, which down-weights easy examples and focuses training on hard-to-classify samples:

$$FL(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t)$$

with $\gamma = 2.0$ and $\alpha = 0.25$. This is particularly beneficial for the Activity Recognition task where class imbalance exists (e.g., Cutting frames are far more frequent than Dropping frames). The loss module also supports standard Cross-Entropy and BCE with logits for binary tasks.

#### Training Stabilisation Callbacks

Three callback mechanisms in the `callbacks/` package prevent overfitting and improve convergence:

1. **Early Stopping** (`callbacks/early_stopping.py`) — Monitors validation loss and stops training after 10 consecutive epochs without improvement (min\_delta = 0.001), preventing unnecessary computation and overfitting.

2. **Model Checkpointing** (`callbacks/checkpoint.py`) — Saves the model state dictionary (`best.pt`) whenever the monitored metric improves. Supports both minimisation (for loss) and maximisation (for accuracy) modes.

3. **ReduceLROnPlateau** (`callbacks/lr_scheduler.py`) — Halves the learning rate (factor = 0.5) when validation loss plateaus for 3 consecutive epochs, with a minimum learning rate floor of 1×10⁻⁶. This adaptive scheduling helps the model escape shallow local minima.

#### Experiment Tracking

All training runs are tracked using **MLflow** with a SQLite backend (`mlflow_tracking.db`). Each run logs:

- Full Hydra configuration as parameters.
- Per-fold, per-epoch metrics: `train_loss`, `train_acc`, `val_loss`, `val_acc`.
- Aggregate metrics: `avg_val_loss` (for k-fold), `val_loss` (for holdout).
- Model artifacts: `best_model.pt` bundle including model state dict, fitted `StandardScaler`, `LabelEncoder`, and full config.
- Registered model entries in the MLflow model registry.

Configuration management is handled by **Hydra**, which enables task switching via CLI overrides (e.g., `python train.py task=knife_sharpness`) and automatic output directory organisation with timestamps.

### 3.4 Evaluation of AI Model

Evaluation uses validation loss and validation accuracy from MLflow-tracked metrics. For k-fold experiments, both average validation loss across all folds and best-fold metrics are reported.

#### Results Summary

| Task | Split | Average / Range | Best Observed |
|---|---|---|---|
| Boning vs Slicing | 10-fold grouped CV | avg\_val\_loss = 0.00963 | val\_loss = 0.00161, val\_acc = **98.68%** |
| Knife Sharpness | 10-fold grouped CV | avg\_val\_loss = 0.21347 | val\_loss = 0.11499, val\_acc = **49.83%** |
| Activity Recognition | Holdout by video\_id | val\_loss = 0.1451 – 0.1873 | val\_acc = **66.22% – 69.64%** |

#### Interpretation

**Boning vs Slicing (Binary):** The near-perfect accuracy (98.68%) and extremely low loss (0.00963 average) confirm that the engineered kinematic features — particularly velocity and acceleration magnitudes — produce highly separable representations for these two activities. The distinct movement patterns of boning (repeated pulling and cutting along bone) versus slicing (smooth lateral movements) are well-captured by the 145-dimensional feature space.

**Activity Recognition (9-class):** Moderate performance (66–70% accuracy) reflects the inherent difficulty of distinguishing nine fine-grained activities from motion capture data alone. Actions like "Reaching" and "Placing/Manipulating" share similar arm motion patterns, and "Idle" periods may vary significantly in posture. The label-specific windowing rules help mitigate class imbalance but cannot fully resolve inter-class similarity.

**Knife Sharpness (3-class):**  

> **Note on saved evaluation results:** The file `models/eval_results_P2.yaml` records `accuracy: 29.73`, which appears anomalous (accuracy values in this pipeline are expected in [0, 1]). This result should be re-validated before use as a headline metric. The discrepancy may stem from an evaluation run using an older pipeline version or a percentage-scale output.

---

## 4. AI Demonstrator

A Streamlit-based demonstrator (`ui_app.py`) is implemented as a single-page application with sidebar navigation across three interactive pages:

### Page 1: Pipeline Architecture & Data Flow

This page provides a visual and interactive trace of the data processing pipeline:

- **Conceptual data flow diagram** — A Graphviz-rendered DAG showing the transformation chain from raw CSV → merged sensors → cleaned labels → filtered labels → engineered features → sanitised features → windowed samples → padded windows → train/val split → scaled tensors → PyTorch DataLoader.
- **Live preprocessing simulation** — Users select a specific CSV file, Video ID, and ML task configuration, then trigger a step-by-step preprocessing trace. The simulation displays:
  - **Stage 1:** Raw row count and sensor stream distribution.
  - **Stage 2:** Sensor merge row reduction and column expansion metrics.
  - **Stage 3:** Label cleaning mapping table showing original dirty strings → cleaned integers.
  - **Stage 3.5:** Task-specific rules showing which labels are excluded and which target column is active.
  - **Stage 4:** Feature engineering with interactive feature plotters, LaTeX formula renders for key algorithms (3D magnitude, kinetic energy, limb energy ratio), and grouped kinematic pair visualisation.
  - **Stage 5:** Windowing output showing generated window count, tensor shape, and post-windowing target class distribution.

### Page 2: Data Preprocessing & EDA

This page enables exploratory data analysis on the raw CSV exports:

- **Flexible data loading** — Two strategies are offered: load top-N rows (1K / 10K / 50K / 100K / All) or filter by specific Video ID for targeted analysis.
- **Data preview** — Full DataFrame view with row count.
- **Summary statistics** — Descriptive statistics (mean, std, min, max, quartiles) for all numeric columns.
- **Missing values audit** — Per-column NaN counts.
- **Visual EDA suite:**
  - Histogram with box plot marginal for distribution analysis of any selected feature.
  - Correlation heatmap (Seaborn) showing inter-feature relationships.
  - Time series line chart (Plotly) tracking any feature across frames, with automatic colour grouping by `video_id` and `sensor_type` to prevent visual artefacts from interleaved data.

### Page 3: Model Training

This page provides a full training launch interface:

- **Task selection** — Dropdown for Boning vs Slicing, Knife Sharpness, or Activity Recognition.
- **Validation strategy** — Auto (uses config default), Holdout, or K-Fold with adjustable split count.
- **Hyperparameters** — Model architecture (BiLSTM / GRU / TCN), epochs, batch size, and learning rate.
- **Custom dataset upload** — ZIP upload with automatic validation of P1/P2/Boning/Slicing directory structure.
- **MLflow integration** — Automatic MLflow UI server launch, status indicator, and direct links to experiment dashboards.
- **Live training monitor** — Streaming log output with:
  - Real-time progress bar tracking fold/epoch advancement.
  - MLflow run ID detection and direct link generation.
  - Phase-aware status messages (data preparation → training → completion).

**Required inputs:** Existing `output_data/` CSVs (or an uploaded dataset ZIP), plus user-selected task and hyperparameters from sidebar controls.

**Produced outputs:** Interactive preprocessing plots, dataset diagnostics, training progress logs with fold/epoch status, and MLflow run links for post-training experiment analysis.

---

## 5. Conclusions

### Final Outcomes

This project delivered a complete end-to-end AI workflow for motion-based activity and sharpness recognition:

1. **A robust preprocessing pipeline** that merges two sensor modalities (velocity and acceleration), cleans 15+ label variants into consistent integers, applies activity-aware label corrections, engineers 145 kinematic features per timestep, and generates padded fixed-length windows ready for sequence modelling.

2. **Strong performance for boning vs slicing classification** — achieving 98.68% best-fold validation accuracy, confirming that engineered kinematic features provide excellent class separability for binary activity tasks.

3. **Moderate performance for full activity recognition** (66–70% accuracy across 9 classes) — a reasonable baseline given the fine-grained nature of the task and the presence of similar-looking activities.

4. **Challenging sharpness classification** (~50% accuracy for 3 classes) — highlighting that motion data alone may be insufficient for inferring tool condition and that this task likely requires additional signal sources or per-cut segmentation.

### Technical Challenges and Resolutions

| Challenge | Resolution |
|---|---|
| **Inconsistent labels across files** | Centralised `LABEL_MAPPING` dictionary mapping 15+ string variants to integers, plus automatic boning Label 5 → 8 override. |
| **Temporal leakage risk** | `StratifiedGroupKFold` and `GroupShuffleSplit` with `video_id` as the group key ensure no windows from the same video appear in both train and validation sets. |
| **Class imbalance** | Focal Loss ($\gamma=2.0$) down-weights easy examples. Label-specific windowing rules for Activity Recognition use smaller strides and skip steps for under-represented classes. |
| **Training instability with noisy features** | `clean_features()` replaces NaN/Inf values. `StandardScaler` fitted on training data only. Gradient clipping set to 1.0. |
| **Reproducibility and experiment tracking** | Hydra manages all configurations with YAML overrides and timestamped output directories. MLflow logs parameters, metrics, and model artifacts for every run. |
| **Sensor alignment** | One-to-one validated inner merge on `(video_id, Frame)` ensures temporal integrity between velocity and acceleration streams. |

### Key Learnings

1. **Data preprocessing and split design are as important as model selection.** The grouped split strategy by `video_id` was essential — without it, validation accuracy would be artificially inflated by frame-level temporal leakage.

2. **Engineered kinematic features can provide high separability for simpler activity tasks.** The 145-feature set combining magnitudes, planar pair magnitudes, total body energy, and energy ratios proved highly effective for binary classification.

3. **More complex targets require richer signal context.** The sharpness classification task showed that the current feature set — designed primarily for activity discrimination — does not capture the subtle biomechanical differences associated with knife condition. Future work could explore per-stroke energy profiles, jerk-based dynamics, or multi-modal sensor fusion.

4. **Modular pipeline design pays off.** Separating configuration (Hydra YAML), training logic (holdout/kfold modules), model definitions, and callbacks into independent modules made it straightforward to run experiments across multiple tasks and strategies without code changes.

5. **Interactive demonstrators accelerate debugging and understanding.** The Streamlit dashboard proved invaluable for verifying preprocessing steps, catching label errors, and communicating pipeline behaviour to non-technical stakeholders.

---

## 6. References

1. Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
2. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *arXiv:1406.1078*.
3. Bai, S., Kolter, J. Z., and Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv:1803.01271*.
4. Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.
5. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019*.
6. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
7. Hydra Documentation. https://hydra.cc/docs/intro/
8. MLflow Documentation. https://mlflow.org/docs/latest/index.html
9. Streamlit Documentation. https://docs.streamlit.io/
10. [If applicable] Motion capture system/vendor documentation used with this dataset.
11. ChatGPT prompt logs used for coding/reporting assistance (include date and prompt text as required by subject policy).

---

## 7. Appendix A

### Source Code and Notebooks

- **GitHub Repository:** [Fill with link]
- **Notebook Folder:** `backend/notebook/`
  - `activity_recognition.ipynb` — Full 9-class activity recognition pipeline.
  - `boning_vs_slicing.ipynb` — Binary boning vs slicing classification.
  - `knife_sharpness.ipynb` — 3-class sharpness classification.
  - `knife_sharpness_segment_position.ipynb` — Sharpness prediction using segment position features.
  - `activity_recognition_cutting_vs_rest.ipynb` — Cutting vs rest binary variant.
  - `activity_recognition_except_4.ipynb` — Activity recognition excluding cutting class.

### Intermediate Data and Model Artefacts

- **Processed CSV exports:** `backend/output_data/` (P1\_boning.csv, P1\_slicing.csv, P2\_boning.csv, P2\_slicing.csv)
- **Trained model checkpoint:** `backend/models/best.pt`
- **Evaluation results:** `backend/models/eval_results_P2.yaml`
- **MLflow experiment database:** `backend/mlflow_tracking.db`
- **MLflow run artefacts:** `backend/mlruns/`

### Run Instructions

```bash
# 1. Generate CSV exports from raw dataset Excel files
python -m data.loader --output-dir output_data

# 2. Train a model (examples for each task)
python train.py task=boning_vs_slicing
python train.py task=knife_sharpness
python train.py task=activity_recognition

# 3. Train with custom overrides
python train.py task=boning_vs_slicing +model.type=gru train.epochs=50 data.batch_size=64

# 4. Evaluate a saved model
python eval.py model_path=models/best.pt data.participant=P2

# 5. Launch the interactive demonstrator
streamlit run ui_app.py

# 6. Launch MLflow UI separately (optional)
mlflow ui --backend-store-uri sqlite:///mlflow_tracking.db --port 5000
```

### Project File Structure

```
backend/
├── configs/
│   ├── train.yaml              # Main training config (Hydra)
│   ├── eval.yaml               # Evaluation config
│   └── task/
│       ├── activity_recognition.yaml
│       ├── boning_vs_slicing.yaml
│       └── knife_sharpness.yaml
├── data/
│   ├── base.py                 # ExcelReader with metadata parsing
│   ├── loader.py               # DataLoader with CSV export
│   ├── preprocessing.py        # Merge, clean, engineer, window
│   └── motion_dataset.py       # PyTorch Dataset wrapper
├── model/
│   ├── bilstm.py               # Bidirectional LSTM architecture
│   ├── gru.py                  # GRU architecture
│   └── tcn.py                  # Temporal Convolutional Network
├── training/
│   ├── core_run.py             # Core training loop (scaler, model, epochs)
│   ├── kfold.py                # K-fold cross-validation orchestrator
│   ├── holdout.py              # Holdout split orchestrator
│   ├── trainer.py              # train_epoch / eval_epoch functions
│   └── loss.py                 # Focal Loss + loss builder
├── callbacks/
│   ├── early_stopping.py       # EarlyStopping callback
│   ├── checkpoint.py           # ModelCheckpoint callback
│   └── lr_scheduler.py         # ReduceLROnPlateau callback
├── utils/
│   ├── seed.py                 # Random seed utility
│   └── device.py               # GPU/CPU device selection
├── train.py                    # Main training entry point
├── eval.py                     # Model evaluation entry point
├── ui_app.py                   # Streamlit demonstrator (679 lines)
├── dataset/                    # Raw Excel files (P1/P2)
├── output_data/                # Processed CSV exports
├── models/                     # Saved model checkpoints
├── mlruns/                     # MLflow experiment artefacts
├── notebook/                   # Jupyter experiment notebooks
└── mlflow_tracking.db          # MLflow SQLite backend
```

- **Public demo link:** [Fill if hosted]
