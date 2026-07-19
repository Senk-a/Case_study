# Commercial Auto Risk Scoring — Case Study Solution

## Overview

This repository contains an improved risk scoring pipeline for commercial auto
insurance policies, covering both **Auto Physical Damage (APD)** and
**Auto Liability (AL)** coverage lines. 

The model outputs a **continuous claim-risk probability per policy**.

## Approach

### Coverage-specific modeling
EDA (boxplots/barplots comparisons and Mutual Information analysis) showed that APD and
AL have materially different risk drivers — several features carried
near-zero individual predictive signal for AL that were informative for APD.
Based on this, **separate models are trained for APD and AL** rather than a
single pooled model, so each can weight its own relevant risk drivers.

At scoring time, the model does **not** output a hard 0/1 label. It produces 
a continuous claim-risk probability per policy,
which is the actual deliverable for downstream use (e.g. risk-based premium
loading).

### Data cleaning and feature decisions
- Dates parsed to proper `datetime`; categorical strings standardized
  (trimmed whitespace, consistent casing) to avoid artificial duplicate
  categories.
- Business-impossible rows removed from training data: `vehicle_count = 0`
  for APD (APD cannot exist without owned vehicles, unlike AL which can cover
  non-owned auto exposure), and negative `annual_premium` values.
- Missing values: general numeric fields imputed to 0; `risk_score_external`
  imputed with conditional group means by coverage type and business type;
  `payment_frequency` missingness encoded as its own explicit category rather
  than imputed, since absence of payment info may itself carry risk signal.
- Features dropped for **data leakage** (post-inception / look-ahead
  information not available at underwriting time): `claim_status`,
  `claim_paid_amount_current_period`, `days_to_first_claim_report`.
- `has_safety_program` dropped because it is absent from `score.csv`.
- `driver_average_age` and `annual_premium` dropped from the **AL model only**,
  based on boxplot/MI evidence of weak individual signal for that coverage
  type.

### Preprocessing
- Right-skewed numeric fields (`prior_loss_amount`, `annual_premium`,
  `prior_year_mileage_000`, `vehicle_count`, `driver_count`) log-transformed,
  then scaled with `RobustScaler` alongside all other numeric
  features — required for a regularized logistic regression.
- Low-cardinality nominal categoricals (`business_type`, `payment_frequency`)
  one-hot encoded.
- High-cardinality `state` target-encoded using scikit-learn's `TargetEncoder`.

### Validation approach
- **Chronological train/test split** (earlier policy years for training, most
  recent year held out for testing) rather than a random split.
- Cross-validation used within `GridSearchCV` for hyperparameter
  tuning (`C`, `l1_ratio`) on the training period only.
- Given severe class imbalance (claims are low-frequency events), model
  selection rely on **probability-based
  metric (ROC-AUC)** rather than accuracy, which
  is misleading when the majority class dominates.

### Model
Logistic regression (`saga` solver) with L1/L2/elastic-net regularization,
chosen for interpretability of coefficients.

## How to run

1. Place `train.csv` and `score.csv` in the `data/`
   folder.
2. Run `01_data_prep_and_training.ipynb` — performs cleaning, EDA-driven filtering, 
   writes prepared score datasets, and builds the preprocessing + model pipeline,
   tunes hyperparameters via cross-validated grid search, fits separate APD
   and AL pipelines, and persists trained model artifacts to `model/`.
3. Run `02_scoring.ipynb` — loads the persisted pipelines and scores `score.csv`
   (2023–2024 policies), producing continuous risk probabilities per policy
   per coverage line. Than it concats sperate tables and writes `data/predictions.csv`.


## Known limitations and trade-offs

- **Threshold kept at the classifier default (0.5)** for any hard-label
  reporting (e.g. confusion matrix). This threshold was not business-tuned
  and should not be read as an operational decision cutoff — the continuous
  probability is the intended output, not the 0/1 label.
- **Target is binary occurrence, not frequency or severity.** The model does
  not distinguish a policy with one claim from one with several, nor does it
  estimate claim size.
- **Feature removal based on univariate Mutual Information** (`driver_average_age`,
  `annual_premium` for AL) may discard features that matter only in
  combination with others, since MI as used here is a marginal, single-feature
  measure. A regularized model could instead be allowed to shrink these
  toward zero rather than removing them outright.
- **No drift monitoring implemented.** New categories like `rideshare` 
  appear in the scoring data but not in training. 
  The model processes them without error, 
  but lacks formal tracking or alerting systems to detect these dataset changes.
- **Highly unbalanced target values**. The dataset suffers from severe class 
  imbalance since insurance claims are naturally zero-inflated, 
  which can limit the classifier's ability to easily isolate rare risk patterns.
- **Reduced training sample size** per model due to coverage segmentation. 
  Splitting the training dataset by `coverage_type` (APD vs. AL) 
  partitions the historical data into separate subsets. 
  While this allows the algorithms to learn specialized feature weights for each risk type, 
  it inherently reduces the total volume of training examples available to fit each individual model pipeline.
