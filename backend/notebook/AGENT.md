# Notebook Plan: Classical ML Baseline for Boning vs Slicing

## Goal

Build a classical machine-learning baseline to classify **boning vs slicing** using window-based statistical features extracted from skeleton joint time-series data.

---

## Data

### Training Data (P1)

* `df_boning` from `P1_boning/output_data`
* `df_slicing` from `P1_slicing/output_data`

Notes:

* Each dataframe already contains the column `activity_type`
* No need to create or modify labels

---

### Test Data (P2)

* `df2_boning` from `P2_boning/output_data`
* `df2_slicing` from `P2_slicing/output_data`

---

## Step 1: Load and Combine Data

* Load P1 boning and slicing dataframes
* Concatenate them into a single dataframe for training
* Verify schema consistency and presence of `activity_type`

---

## Step 2: Train / Validation Split

* Split P1 data into training and validation sets using:

  * `train_test_split`
  * `test_size = 0.2`
  * `stratify = activity_type`
  * fixed `random_state`

---

## Step 3: Stage 1 Feature Pruning (Pre-Aggregation, Unsupervised)

**Applied on raw frame-level features (train set only)**

* Remove constant columns
* Remove near-zero variance columns (`VarianceThreshold`)
* Optionally remove obviously irrelevant joints using domain knowledge

Apply the same column pruning to validation and P2 test sets.

---

## Step 4: Temporal Windowing

* Create sliding windows over the time dimension using:

  * Window size = **30 frames** (0.5 seconds at 60 FPS)
  * Explicitly define stride (e.g. non-overlapping or fixed overlap)

---

## Step 5: Window-Based Feature Extraction

For each window and each remaining signal, compute statistical features such as:

* Mean

* Standard deviation

* Minimum

* Maximum

* Peak value

* Trapezoidal integral

* Flatten all window-level features into a tabular dataset

---

## Step 6: Stage 2 Feature Pruning (Post-Aggregation)

**Applied on window-level features (train set only)**

* Remove low-variance window features

Apply the same removals to validation and test sets.

---

## Step 7: Multicollinearity Analysis

* Compute correlation matrix on **training window-level features**
* Visualize correlations using a heatmap

---

## Step 8: Remove Highly Correlated Features

* Identify highly correlated feature pairs (e.g. |r| > 0.9)
* Retain one feature per correlated group
* Apply the same feature removals to validation and test sets

---

## Step 9: ANOVA Feature Selection

* Apply ANOVA F-test (`SelectKBest(f_classif)`) on **training window-level features only**
* Select the top **20 features**
* Transform validation and test sets using the fitted selector

---

## Step 10: Visualize ANOVA Results

* Plot ANOVA F-scores for all candidate window-level features
* Highlight and label the top 20 selected features

---

## Step 11: Normalization

* Fit a scaler (e.g. `StandardScaler`) on training window features
* Apply the same scaler to validation and P2 test window features

---

## Step 12: Model Training

Train the following models using the selected window-based features:

* Decision Tree (with constrained maximum depth)
* Random Forest
* Extra Trees (or another ensemble-based classifier)

---

## Step 13: Validation Evaluation

Evaluate models on the validation set using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Compare model performance.

---

## Step 14: Testing on P2

* Evaluate trained models on the P2 test set
* Report the same performance metrics
* Compare generalization between P1 validation and P2 testing

---

## Step 15: Model Interpretability

* Visualize and print the trained Decision Tree with limited depth
* Identify key features contributing to boning vs slicing classification

---

## Step 16: Final Analysis

Discuss:

* Most informative features selected by ANOVA and tree-based models
* Impact of two-stage feature pruning on dimensionality and performance
* Effectiveness and limitations of window-based statistical features
* Motivation for using temporal deep-learning models (e.g. TCN) as a next step
