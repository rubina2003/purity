# Water Potability Project: Findings and Recommended Approach

Summary of the research, current decisions, risks, and next steps for a Nepal-focused water-potability classification project using Logistic Regression, Random Forest, and SVM. The project is still in the research phase: no datasets have been downloaded and no code or models have been implemented yet.

> **Document status (scope cleanup, November 2026):** This document records the **preliminary research stage** of the project and is preserved for academic traceability. Several decisions recorded here have since been revised. Where this document conflicts with the root `README.md`, the README is authoritative. The main superseded items are:
> - **Scikit-learn supporting utilities (sections 7, 8, 12):** superseded. Scikit-learn is strictly prohibited for all purposes, including splitting, cross-validation, metrics, preprocessing, and persistence. Seaborn is not part of the approved stack.
> - **"Do not add a web application" (section 7):** superseded. Purity is a web-based system (React.js + Vite, Python Flask, MySQL).
> - **Rule-based labeling of Nepali datasets and NDWQS relabeling of Kaggle (sections 3, 4, 6, 11):** not part of the approved implementation. The `label_ndwqs.py` script referenced below was never created in the workspace.
> - **Linear-only SVM (sections 8, 12):** revised. The approved SVM implementation includes Linear and RBF kernel calculations.
> - **Provisional folder structure (section 7):** superseded by the structure documented in the README.
>
> The finalized scope uses one dataset (Kaggle `water_potability.csv`), nine physicochemical features, the existing binary `Potability` target, three custom classifiers, no synthetic training records, no K-Means clustering, and no WQI-based categories. Parameter-level comparison against NDWQS 2079 is a **separate** standards-based assessment, not a labeling method or an ML algorithm.

Companion files:
- `water_potability_datasets.md`: dataset links, descriptions and columns
- `label_ndwqs.py`: script that labels samples using NDWQS limits *(planned during research; never created and not part of the approved implementation)*

---

## 1. Problem framing

- **Task type:** Supervised binary classification (potable = 1, non-potable = 0). Each sample has input features (pH, TDS, turbidity, etc.) and a target label.
- **Where labels come from matters:**
  - Kaggle: labels are provided.
  - Nepali data: raw measurements only. Labels must be created from standards (rule-based), after which training is standard supervised learning.
- **Not applicable here:** Clustering (unsupervised) and training on one dataset to label another (pseudo-labeling / semi-supervised). See section 5.

---

## 2. Key finding: data availability

- There is **no large, open Nepali dataset with potability labels**.
- Available Nepali data is small and unlabeled (measurements only).
- The only dataset with real labels is the **Kaggle Water Potability** dataset (3,276 samples, 9 features, binary target), whose region and collection method are not documented.

### Dataset overview

| Dataset | Region | Labeled? | Size | Notes |
|---|---|---|---|---|
| Kaggle Water Potability | Unspecified | Yes | 3,276 | Only independently labeled set; weak signal, missing values |
| KUKL Kathmandu groundwater (PANGAEA) | Kathmandu Valley | No | 60 sources | CC BY 4.0; best open Nepal source |
| Open Data Nepal water samples | Kathmandu, Pokhara, Butwal, Birgunj, Narayanghat | No | Many tiny files | Mostly surface water |
| REACH-Nepal Catalyst (Oxford) | Western Nepal | No | 120 households | Includes microbial data; column list not verified |
| NAST drinking water study | Nationwide | No | 250 samples | Paper only; raw data needs author request |
| Terai arsenic testing (DWSS) | Terai | No | Thousands of wells | Not openly downloadable; request needed |
| CWC surface water quality | India | No | Large (1963-2021) | Open government license |
| CPCB / GEMStat | India / global | No | Large | Useful for other regions |

Links are in `water_potability_datasets.md`.

### Known issues with the Kaggle dataset

- `ph` has 491 missing values; `Sulfate` and `Trihalomethanes` also have gaps.
- Classes are imbalanced (roughly 39% potable).
- Feature-label correlations are very weak, so expect modest accuracy.

---

## 3. Labeling the unlabeled data

> **Status:** Preliminary research proposal. Purity does not label external datasets; it uses only the existing `Potability` labels in the Kaggle dataset. The NDWQS limits listed below remain useful as unverified reference material for the separate standards-based assessment, subject to verification against the official NDWQS 2079 text.

Use a script, not manual labeling. The rule: **if any measured parameter exceeds its limit, label the sample non-potable.**

### NDWQS limits used (mg/L unless noted)

| Parameter | Limit |
|---|---|
| Turbidity | 5 NTU |
| pH | 6.5 to 8.5 |
| Total dissolved solids | 1000 |
| Conductivity | 1500 uS/cm |
| Iron | 0.3 |
| Manganese | 0.2 |
| Arsenic | 0.05 (WHO guideline is 0.01) |
| Fluoride | 0.5 to 1.5 |
| Ammonia | 1.5 |
| Chloride | 250 |
| Sulfate | 250 |
| Nitrate | 50 |
| Total hardness | 500 |
| E. coli | 0 per 100 mL |

**Verify before publishing.** These were compiled from a web copy of the NDWQS table whose formatting was jumbled, so values were matched to parameters by order. A newer 2022 edition of the standards is referenced in the literature and may differ from the 2005 values.

### Script behavior (`label_ndwqs.py`)

- Standardizes column names and reports unrecognized columns.
- Outputs `potable` (1 / 0 / blank if too few parameters measured), `violations`, and `n_params_checked`.
- Options: `--who-arsenic` (stricter arsenic limit) and `--min-params` (minimum measured parameters to call a sample potable).

---

## 4. Risks and how to handle them

### 4.1 Rule-based labels are only as good as the measured parameters

A sample can only fail on parameters that were measured. A dataset without arsenic can never fail on arsenic, so identical water may get different labels in different datasets. The label really means "no violation found among measured parameters."

Mitigations:
- Define a **core panel** of parameters shared across datasets (pH, TDS or conductivity, turbidity, hardness, plus nitrate or ammonia where available) and label on that panel.
- Keep a second, full-panel label where more parameters exist, and record which parameters each label used.
- Do not pool datasets labeled on different bases.
- Leave samples with too few measured parameters unlabeled.

### 4.2 Circularity

If labels are a threshold rule on the features, the models just rediscover the rule and accuracy is meaningless. Avoid this by building the label from lab-only parameters (arsenic, iron, ammonia, nitrate, chloride) and training on field-measurable features (pH, temperature, DO, hardness, and so on).

### 4.3 KUKL class-balance warning

KUKL reports arsenic from 0.2 to 1.5 mg/L. Every value in that range exceeds the 0.05 mg/L limit, so if the units are correct, **every sample would be labeled non-potable** and no classifier can be trained. Check the units in the methods PDF on the PANGAEA page. If the units are right, drop arsenic from the label or choose a different target (for example iron or ammonia exceedance).

### 4.4 Small samples

KUKL has only 60 sources. Use repeated stratified or leave-one-out cross-validation and report confidence intervals.

---

## 5. Why not pseudo-label the Nepali data with a Kaggle-trained model

- **Feature mismatch:** Kaggle needs chloramines, trihalomethanes, organic carbon and sulfate, which most Nepali datasets lack. Only a few columns are shared (for KUKL, essentially pH and hardness).
- **Domain shift:** The Kaggle data has no documented region or source type, so there is no reason it transfers to Kathmandu groundwater or Nepali rivers.
- **Weak signal:** Noisy predictions would become noisy labels.
- **Circularity:** Training on labels produced by another model recreates that model.
- Once both datasets use the same rule, the rule can label Nepali data directly, so no model is needed for labeling.

A Kaggle-to-Nepal test on the few shared columns is still possible as an optional domain-shift analysis, not as a labeling method.

---

## 6. Should Kaggle be relabeled with NDWQS?

**Verdict: not as the foundation; run a quick check first and keep it as an optional second label.**

> **Finalized decision:** Kaggle is **not** relabeled with NDWQS thresholds in Purity. The original `Potability` labels are the only ML target. NDWQS comparisons are performed separately as a parameter-level standards assessment and are never used as training labels.

Pros: gives a consistent label definition across datasets.

Cons:
- Kaggle covers only part of the standard (no arsenic, iron, ammonia, nitrate, chloride, E. coli), so the new label means "passes the checkable subset."
- Most rows may fail. One analysis describes dissolved solids peaking near 20,000 while the TDS limit is 1000, and sulfate may often exceed 250. Verify these in your copy. If almost every row fails, you are left with nearly one class.
- It replaces the only independent label with a threshold rule, which leads back to circularity and inflated accuracy.
- It does not fix the cross-dataset pseudo-labeling problem.

Quick viability check:

```python
import pandas as pd
df = pd.read_csv("water_potability.csv")
ok = df.ph.between(6.5, 8.5) & (df.Solids <= 1000) & (df.Conductivity <= 1500) \
     & (df.Turbidity <= 5) & (df.Hardness <= 500) & ((df.Sulfate <= 250) | df.Sulfate.isna())
print(ok.value_counts())
```

If the split is reasonable, store it as `potable_ndwqs` alongside `potable_original`. If almost everything fails, skip it.

---

## 7. Current project constraints and scope decisions

These decisions reflect the current planning discussion. They are project constraints, not claims that the implementation has already been completed.

> **Status:** The constraints in this section reflect the preliminary planning discussion. The items marked *Superseded* below have been replaced by the finalized decisions documented in the root `README.md`.

- **Current stage:** Research only. No dataset has been downloaded, no code has been written, and none of the three models has been implemented or trained.
- **Primary goal:** Establish a transparent, reproducible baseline and compare three classifiers: Logistic Regression, Random Forest, and SVM.
- **Implementation requirement:** Implement the core learning/training logic for all three algorithms from scratch. Scikit-learn may be used for supporting utilities such as data splitting, cross-validation, and evaluation metrics, but not for its built-in Logistic Regression, Random Forest, or SVM model implementations. Confirm any final interpretation against the college's exact rules. *(Superseded: Scikit-learn is strictly prohibited for every purpose. Data splitting, preprocessing, cross-validation, evaluation metrics, and persistence are all implemented manually or with the Python Standard Library.)*
- **Permitted supporting tools (subject to course rules):** pandas and NumPy for data handling and numerical calculations; Matplotlib and Seaborn for visualization; selected scikit-learn utilities for supporting tasks. *(Superseded: the approved supporting libraries are NumPy, Pandas, Matplotlib, and the Python Standard Library. Seaborn and Scikit-learn are not used.)*
- **Scope control:** Start with one baseline dataset. Do not combine datasets, add a web application, or expand the number of algorithms unless a clear project need emerges. *(Partially superseded: the single-dataset and three-algorithm constraints remain; however, Purity is now a web-based system built with React.js, Flask, and MySQL.)*
- **Documentation priority:** Record dataset provenance, feature definitions and units, missing measurements, features absent entirely, label meaning, assumptions, limitations, preprocessing decisions, and experiment results.
- **Visualization priority:** The initial data audit should produce visualizations for target-class balance, missingness, feature distributions, outliers or unusual values, feature relationships/correlations, and later model evaluation.

### Recommended project structure (provisional)

```text
water-potability-project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_data_exploration.ipynb
│   └── 03_model_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── svm.py
│   └── evaluation.py
├── docs/
│   ├── dataset_card.md
│   ├── data_limitations.md
│   └── experiment_log.md
└── results/
    ├── figures/
    └── model_comparison.csv
```

This is a suggested structure, not a requirement to create every file immediately. A custom Random Forest will need a decision-tree implementation as a building block. *(Superseded: the project structure is now defined in the root `README.md`.)*

---

## 8. Baseline design and fair model comparison

The baseline should use one selected labeled dataset and its original target labels, provided the dataset's source documentation is sufficiently clear. The Kaggle Water Potability dataset is the leading candidate, but it must be downloaded and audited before the choice is treated as final.

For a fair comparison, all three models should use the same eligible input features, target definition, train/test split, cross-validation folds, and reported evaluation metrics. Preprocessing must be fitted using training data only to avoid data leakage. The held-out test set should not be used repeatedly to choose models or tune settings.

Recommended initial evaluation:

- Stratified train/test split for a held-out final evaluation.
- Stratified 5-fold cross-validation on training data for model comparison and tuning, if sample size and class counts permit. *(Purity note: the approved methodology evaluates all models on one reproducible stratified train-test split; any cross-validation, if added, must be implemented manually.)*
- Report accuracy, precision, recall, F1-score, ROC-AUC when appropriate, and confusion matrices.
- Explicitly inspect recall and false positives for the non-potable class, because predicting non-potable water as potable is a safety-relevant error.
- Report variability across folds where possible; do not present a single score as definitive.
- Do not assume a target threshold of 0.5 is optimal for every safety-oriented use case. Any threshold selection must use training/validation data, not the final test set.

Do not promise a specific accuracy range in advance. Performance must be measured and interpreted from the actual experiment.

### From-scratch implementation plan

1. **Logistic Regression:** implement the sigmoid, loss, gradients/optimization, training loop, and prediction.
2. **Decision Tree building block:** implement a split criterion such as Gini impurity, feature/split search, recursive tree construction, stopping conditions, and prediction.
3. **Random Forest:** implement bootstrap sampling, randomized feature selection, multiple trees, and aggregation of tree predictions.
4. **SVM:** start with a clearly scoped linear SVM and implement its objective/optimization and prediction. Only add a nonlinear kernel if it is required and feasible within the project scope. *(Revised: the approved SVM implementation includes manual Linear and RBF kernel calculations.)*
5. **Evaluation:** supporting evaluation utilities may be used if permitted; otherwise implement the required metrics and validate them against small known examples. *(Superseded: all evaluation metrics — Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix — are implemented manually; no Scikit-learn utilities are used.)*

Keep the implementations simple, testable, and documented. The goal is a defensible comparison, not recreating every feature offered by a production machine-learning library.

---

## 9. Dataset audit, visualizations, and documentation

The first implementation deliverable should be a dataset audit, not a trained model. Do not start model training until the dataset has been downloaded, its schema inspected, and the most important data-quality issues recorded.

### Dataset card

Record at least:

- Exact dataset name, source link, access date, license, and original filename.
- Number of rows and columns.
- Target column and documented meaning of each target value.
- Feature names, definitions, units, and plausible ranges when documented.
- Geographic origin, source-water type, sampling process, and laboratory/field methods—or explicitly record them as unknown or undocumented.
- Duplicate rows, missing-value counts and percentages, data types, and potential unit inconsistencies.
- Whether each concern is confirmed, suspected, or still unverified.

### Missing measurements versus absent features

Keep these two cases separate:

- **Missing measurement:** A feature exists in the dataset, but a sample has no recorded value. Imputation may be considered after the missingness pattern is inspected. Any imputer must be fitted on training data only.
- **Absent feature:** The dataset never collected or contains no column for that parameter. Imputation cannot recover a feature that was never measured. Record it as unavailable and do not silently invent a value.

Create a feature-availability table showing each feature, definition/unit, missing count and percentage, whether it is present in other candidate datasets, and the planned handling.

### Initial visualizations

Create and preserve figures for:

1. Target-class counts and percentages.
2. Missingness by feature (counts and percentages).
3. Histograms or distribution plots for numerical features.
4. Box plots to inspect unusual values and compare feature distributions by target class.
5. A feature-correlation heatmap.
6. Selected feature-versus-target plots where they help answer a specific question.

Later, add model comparison plots and confusion-matrix heatmaps. Visual patterns are exploratory evidence, not proof of causation or model usefulness. Investigate unusual values before removing them.

### Data limitations register

Maintain `docs/data_limitations.md` with a row or entry for each issue. Suggested fields are: ID, finding, evidence/source, status (confirmed/suspected/unverified), possible impact, decision/action, and follow-up. Examples to investigate include undocumented geographic origin, missing measurements, entirely absent parameters, unclear label methodology, class imbalance, potential unit problems, and small sample sizes in local datasets. Do not mark an issue as confirmed until the data or documentation supports it.

---

## 10. Dataset variation and generalization risk

**Concern:** A model may perform reasonably on the dataset used for training but perform poorly on samples from a different region, water-source type, measurement process, or label definition. This is commonly described as domain shift.

Differences that must be investigated before comparing or combining datasets include:

- Feature availability and feature definitions.
- Measurement units and measurement methods.
- Source-water types and geographic coverage.
- Sampling procedures and time periods.
- Target-label definitions and the evidence used to assign labels.
- Class balance and population differences.

### Decision: keep the baseline and Nepal investigation separate

- **Baseline experiment:** Start with one labeled dataset (Kaggle is the leading candidate) and compare the three algorithms on that dataset. State clearly that results apply to the evaluated dataset and protocol, not automatically to all water sources or Nepal.
- **Nepal investigation:** Audit Nepali datasets separately. Determine which measurements, units, sampling details, and credible labels are available before deciding whether a local classification experiment is feasible.
- **Do not pool datasets by default.** Combine them only after documenting compatible features, units, measurement methods, populations, and target definitions. Matching column names alone is insufficient.
- **Do not use Kaggle model predictions as ground-truth labels for Nepali samples.** Such pseudo-labels can carry the original model's errors and do not solve domain shift.
- **Optional later work:** If a Nepali dataset has compatible features and credible labels, consider a clearly separated external-validation/domain-shift analysis. Do this only if the sample size and class distribution support meaningful evaluation.

The project does not need to prove that a model works everywhere. It should be explicit about what the available data can and cannot support.

---

## 11. Nepal-specific data and labeling: unresolved research tasks

The Nepal case study remains a possible second phase, not a prerequisite for the initial baseline. Before creating local labels, resolve the issues below.

> **Finalized decision:** A Nepal-specific labeling or classification experiment is **outside** the approved Purity scope. The open questions below are preserved as research notes; only the NDWQS verification item remains relevant, because verified limits are needed for the separate standards-based assessment.

- Verify the applicable NDWQS edition and each parameter limit/units against an authoritative standard.
- Confirm the KUKL arsenic units from the original PANGAEA methods documentation before using those values to assign labels.
- Record exactly which parameters were measured for each sample and which parameters were used to derive any label.
- Avoid pooling labels derived from different parameter panels. A sample can only be assessed against the parameters actually measured.
- Avoid circularity: if a target is defined by thresholding a feature, training a model on that same feature can simply reproduce the threshold rule. A field-measurement-versus-laboratory-compliance setup may be more informative if the data supports it.
- If all samples have one class, a binary classifier cannot be meaningfully trained/evaluated on that dataset. Do not alter labels just to create class balance. Investigate units, label definition, or another justified target.
- Request additional data only after the initial dataset audit establishes what is missing and what additional data would contribute.

Do not describe rule-derived labels as independent ground truth. They represent compliance with the chosen rules for the parameters measured, not necessarily comprehensive drinking-water safety.

---

## 12. Current next steps

Proceed in this order and keep the scope small:

> **Status:** This was the preliminary next-steps list. The approved 12-phase development roadmap in the root `README.md` replaces it. Items 1, 9, and 11 are superseded as noted.

1. [ ] Confirm the college's implementation rules; current working interpretation is option B: implement the core algorithms ourselves, while using permitted supporting utilities. *(Superseded: the finalized rule prohibits Scikit-learn for every purpose; all algorithms and supporting operations are implemented manually.)*
2. [ ] Confirm the exact Kaggle dataset source and license, then download the original file without modifying it.
3. [ ] Create a dataset card with source, target meaning, feature definitions/units, provenance, and unknowns.
4. [ ] Run the first audit: shape, data types, duplicate rows, missingness, class counts, summary statistics, and suspicious values.
5. [ ] Save the initial visualizations and create the data limitations register.
6. [ ] Decide the fixed feature set, target definition, split strategy, cross-validation approach, and evaluation metrics before comparing models.
7. [ ] Implement and test Logistic Regression from scratch.
8. [ ] Implement and test a decision tree, then extend it into Random Forest.
9. [ ] Implement and test a linear SVM. *(Revised: implement and test a custom SVM with Linear and RBF kernels.)*
10. [ ] Compare the three models fairly and document results, errors, uncertainty, and limitations.
11. [ ] Revisit Nepal-specific datasets only after the baseline experiment is established and the local-data labeling questions are addressed. *(Superseded: Nepal-specific datasets are background context only and are outside the approved ML scope.)*

### Open items carried forward

- [ ] Confirm KUKL arsenic units from the PANGAEA methods PDF.
- [ ] Verify NDWQS limits against the authoritative standard (including whether the 2005 or 2022 edition applies).
- [ ] Inspect the actual Kaggle dataset before deciding whether optional NDWQS-based relabeling is viable.
- [ ] Request raw NAST, Terai arsenic, or REACH-Nepal data only if needed and access is feasible.
- [ ] Decide whether a Nepal-specific experiment is feasible after documenting the available features, labels, sample sizes, and measurement methods.
