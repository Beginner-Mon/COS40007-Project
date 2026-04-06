# Backend README

## What this backend does
This backend is a PyTorch + Hydra activity-recognition pipeline for motion data.

It:
- reads raw motion-capture Excel files,
- converts them into model-ready tabular data,
- creates temporal windows,
- trains sequence models (BiLSTM/GRU),
- saves artifacts/checkpoints,
- and evaluates trained models on selected participants.

Main entry points:
- `train.py` for training
- `eval.py` for evaluation

---

## How the backend uses the `dataset/` folder

### 1) Raw dataset structure
Raw files are expected under:
- `dataset/P1/Boning/*.xlsx`
- `dataset/P1/Slicing/*.xlsx`
- `dataset/P2/Boning/*.xlsx`
- `dataset/P2/Slicing/*.xlsx`

Each Excel file is parsed from these sheets:
- `Segment Velocity`
- `Segment Acceleration`
- `Markers`

### 2) Raw Excel parsing and labeling
`data/base.py` (`ExcelReader`) does the following:
- reads sensor sheets (`Segment Velocity`, `Segment Acceleration`),
- reads `Markers` sheet and converts frame ranges to labels,
- fills/normalizes `Label` values,
- injects metadata columns:
  - `person_id`
  - `activity_type`
  - `knife_sharpness_score`
  - `sensor_type`
  - `video_id`

### 3) Dataset loading and CSV export
`data/loader.py` loads all `.xlsx` files by person/activity, validates feature columns, concatenates DataFrames, and can save CSV outputs.

Default CSV files are stored in `output_data/`:
- `P1_boning.csv`
- `P1_slicing.csv`
- `P2_boning.csv`
- `P2_slicing.csv`

`Markers` sheet is still required for labeling.

#### Sheet selection behavior
- If `--sheets` is omitted, loader processes default sheets only:
  - `Segment Velocity`
  - `Segment Acceleration`
- If `--sheets` is provided, loader processes only the provided sheet list (exclusive override).
- Provided sheets are not merged with default sheets.

#### Commands
- Default behavior (two default sheets):
  - `python -m data.loader`
- Process one sheet only:
  - `python -m data.loader --sheets "Segment Velocity"`
- Process custom sheets only:
  - `python -m data.loader --sheets "Custom Sheet A" "Custom Sheet B"`

#### Output filename rules
- Default sheet mode (`Segment Velocity` + `Segment Acceleration`):
  - `P1_boning.csv`, `P1_slicing.csv`, `P2_boning.csv`, `P2_slicing.csv`
- Custom/non-default sheet mode:
  - `P1_boning_[sheetname].csv`
  - `P1_slicing_[sheetname].csv`
  - `P2_boning_[sheetname].csv`
  - `P2_slicing_[sheetname].csv`

`[sheetname]` is normalized to lowercase with non-alphanumeric characters replaced by `_`.

### 4) Preprocessing used for modeling
`data/preprocessing.py` provides:
- fixed feature contract (`FEATURE_COLS`),
- window generation by `video_id` (`create_windows`),
- optional activity-specific label remapping (`apply_activity_label_overrides`),
- NaN/inf cleanup (`clean_features`),
- normalization with `StandardScaler` (`normalize_features`).

`train.py` then:
- loads CSVs from `output_data/`,
- filters by `sensor_type` (default: `Segment Velocity`),
- splits by `video_id` (not by frame) to reduce leakage,
- windows the sequences,
- can apply task-scoped activity overrides (for example, boning `Label=5 -> 8`),
- can apply task-scoped label-specific window rules (`window_size`, `window_stride`, `frame_skip_step`),
- fits scaler on train windows and applies to val windows,
- label-encodes targets,
- trains BiLSTM using config in `configs/train.yaml`,
- saves run outputs in `outputs/...` and shared artifacts in `artifacts/shared/`.

### Task-scoped dynamic window config
Dynamic windows are optional and configured per task YAML under `configs/task/`:

- `activity_label_overrides.enabled`: enable/disable activity-specific remaps.
- `activity_label_overrides.by_activity`: per-activity remap dict.
- `windowing.label_specific_rules`: per-label target rules.

Global defaults remain under `configs/train.yaml`:

- `data.window_size`
- `data.stride`
- `data.frame_skip_step`
- `data.pad_target_len`

`eval.py`:
- loads model checkpoint + saved scaler/label encoder,
- applies the same preprocessing steps,
- evaluates on configured participant data,
- writes evaluation results (e.g., `eval_results_*.yaml`).

---

## What was done to the dataset in `notebook/activity_recognition.ipynb`
The notebook performs an experimental end-to-end dataset preparation and training flow.

### Data loading and filtering
- Loads `P1_boning.csv`, `P1_slicing.csv`, `P2_boning.csv`, `P2_slicing.csv` from `output_data/`.
- Filters to `sensor_type == "Segment Velocity"`.
- Uses P1/P2 data for analysis and model development with explicit video-based hold-out selection.

### Label cleanup/relabeling
- Standardizes inconsistent label strings to numeric class IDs.
- Converts labels to numeric/int64 for PyTorch targets.
- Applies a boning-specific adjustment where label `5` is remapped to `8`.

### Feature engineering
- Starts from raw motion feature columns from `data.preprocessing.FEATURE_COLS`.
- Adds engineered motion features such as:
  - per-joint speed,
  - total body energy,
  - mean/variance of joint speed.
- Uses `USE_FEATURE_ENGINEERING` flag to switch between raw-only and raw+engineered feature sets.

### Normalization and split strategy
- Holds out specific `video_id` values for validation (leakage-aware split).
- Fits `StandardScaler` on train frames only.
- Applies the trained scaler to validation frames.

### Windowing
- Builds temporal windows with:
  - `WINDOW = 60`
  - `STRIDE = 30`
- Assigns each window label by majority vote over frame labels in that window.

### Augmentation and encoding
- Encodes labels via `LabelEncoder`.
- Includes optional jitter-based augmentation (`USE_AUGMENTATION`) for training windows.

### Training/evaluation in notebook
- Defines `ActivityBiLSTM` model.
- Uses Focal Loss (`gamma=2.0`) for class imbalance.
- Trains with Adam, gradient clipping, ReduceLROnPlateau, and early stopping.
- Saves best checkpoint to `notebook/best_bilstm.pt`.
- Evaluates on held-out validation videos.

---

## Related files
- `data/base.py`
- `data/loader.py`
- `data/preprocessing.py`
- `train.py`
- `eval.py`
- `configs/train.yaml`
- `configs/eval.yaml`
- `notebook/activity_recognition.ipynb`
