# Activity Recognition (BiLSTM - Time Series)


## Pipeline

### 1. Exploratory Data Analysis (EDA)

#### Goals:
- Check label continuity
- Detect noisy annotations

#### Steps:
- Plot labels over frame index
- Compute:
  - Segment length (min / max / mean)
  - Class distribution

#### Expected Outcome:
- Determine whether label smoothing is needed
- Estimate appropriate window sizes

---

### 2. Data Preprocessing


#### 2.2 Temporal Downsampling

Current:
- 60 FPS → skip = 2 → ~30 FPS

Experiments:
- skip = 1 (no downsampling)
- skip = 2 (baseline)
- skip = 3 / 5

---

#### 2.3 Feature Normalization

Normalize each feature:
```python
x = (x - mean) / std