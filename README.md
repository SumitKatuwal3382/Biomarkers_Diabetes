# Biomarker Discovery for Diabetic Retinopathy Using Machine Learning on Gene Expression Data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SumitKatuwal3382/Biomarkers_Diabetes/blob/main/Biomarkers_test_2.ipynb)

---

## Abstract

Diabetic Retinopathy (DR) is a microvascular complication of diabetes mellitus and a leading cause of preventable blindness worldwide. Early molecular detection remains a critical unmet clinical need. This study presents a machine learning pipeline applied to high-dimensional gene expression data to: (1) identify statistically discriminative gene biomarkers that distinguish DR patients from healthy controls, and (2) evaluate the classification performance of multiple supervised learning models. Using a combined dataset of 195 patient samples and 9,432 gene features, we applied ANOVA-based feature selection, class imbalance correction via SMOTE, and compared seven classifiers — including baseline models and advanced approaches such as CatBoost, SVM with RBF kernel, Bayesian-optimized XGBoost, and a stacking ensemble. The baseline XGBoost model achieved the highest hold-out test accuracy of **84.62%**, while advanced models demonstrated stronger cross-validation generalization, with the Optuna-tuned XGBoost reaching **87.00% CV accuracy**. Key candidate biomarker genes identified include `PIEZO1`, `HIF1A-AS3`, `PLXNB2`, `RPS2P5`, and `RNF144B`.

---

## Table of Contents

1. [Background](#background)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
   - [Preprocessing](#1-preprocessing)
   - [Feature Selection](#2-feature-selection--anova-f-test)
   - [Class Imbalance Handling](#3-class-imbalance-handling--smote)
   - [Baseline Models](#4-baseline-models)
   - [Advanced Models](#5-advanced-models)
4. [Results](#results)
   - [Baseline Results](#baseline-model-results)
   - [Advanced Results](#advanced-model-results)
   - [Full Comparison](#full-model-comparison)
   - [Per-Class Performance](#per-class-performance-advanced-models)
5. [Candidate Biomarkers](#candidate-biomarker-genes)
6. [Visualizations](#visualizations)
7. [Discussion](#discussion)
8. [Reproducibility](#reproducibility)
9. [Repository Structure](#repository-structure)

---

## Background

Diabetic Retinopathy affects approximately one-third of all people with diabetes and is caused by progressive damage to the blood vessels of the retina. Clinical diagnosis relies on ophthalmoscopic imaging, which is expensive and requires specialist interpretation. Molecular biomarkers derived from gene expression profiling offer a complementary diagnostic route — one that could be implemented as a blood test and enable earlier, cheaper, and more scalable screening.

Gene expression data is inherently high-dimensional (thousands of genes per patient) and biologically noisy, making it a challenging but important target for machine learning. The key tasks are: (1) dimensionality reduction to find the subset of genes most associated with disease status, and (2) building classifiers robust enough to generalize to unseen patients despite small sample sizes.

---

## Dataset

| Property | Value |
|---|---|
| File | `FINAL_COMBINED_DATASET.csv` |
| Total samples | 195 patients |
| Gene expression features | 9,432 genes |
| Class — DR (Diabetic Retinopathy) | 125 samples (64.1%) |
| Class — Control (Healthy) | 70 samples (35.9%) |
| Class imbalance ratio | ~1.79 : 1 (DR : Control) |

The dataset is a combined expression matrix where each row is a patient sample and each column (except `Label`) is a gene. The `Label` column takes values `DR` or `Control`.

**Note on class imbalance:** The dataset is moderately imbalanced. Models trained without correction will be biased toward predicting DR (the majority class). This is addressed via SMOTE oversampling (see §3).

---

## Methodology

### 1. Preprocessing

- Target label encoding: `DR → 1`, `Control → 0`
- Stratified 80/20 train-test split to preserve class proportions
  - Training set: **156 samples** (100 DR, 56 Control)
  - Hold-out test set: **39 samples** (25 DR, 14 Control)
- `random_state=42` used throughout for full reproducibility

### 2. Feature Selection — ANOVA F-Test

With 9,432 gene features and only 195 samples, direct model training would be severely underdetermined. We applied **Analysis of Variance (ANOVA F-test)** via scikit-learn's `SelectKBest` to rank genes by their statistical discriminability between classes.

- **Method:** One-way ANOVA F-test between DR and Control groups for each gene
- **Selection:** Top **k = 50** genes retained
- **Fit:** Selector fitted only on training data, then applied to test data to prevent data leakage

This reduces the feature space from 9,432 to 50 — a **99.5% reduction** — while retaining the most biologically relevant signal.

**Top 10 genes selected by ANOVA F-score:**

| Rank | Gene | Biological Relevance |
|---|---|---|
| 1 | NEK6 | Serine/threonine kinase involved in mitotic regulation |
| 2 | KCNE3 | Potassium channel subunit; linked to ion transport dysregulation |
| 3 | RNPEPL1 | Aminopeptidase; implicated in protein processing |
| 4 | PLXNB2 | Plexin receptor; involved in vascular development and angiogenesis |
| 5 | MSRB1 | Methionine sulfoxide reductase; oxidative stress response |
| 6 | Y_RNA | Non-coding RNA; roles in DNA replication and stress response |
| 7 | DUSP1 | Dual specificity phosphatase; key regulator of MAPK pathway |
| 8 | PRAM1 | PML-RARA regulated adaptor molecule |
| 9 | STEAP4 | Metalloreductase; linked to inflammation and insulin signaling |
| 10 | EGR1 | Early growth response transcription factor; angiogenesis and hypoxia |

### 3. Class Imbalance Handling — SMOTE

The original training set has 100 DR and 56 Control samples. Without correction, models overfit to the majority class. We applied **Synthetic Minority Oversampling Technique (SMOTE)**:

- Generates synthetic Control samples by interpolating between real minority-class samples in the 50-gene feature space
- Result: **200 balanced training samples** (100 DR, 100 Control)
- SMOTE applied **after** feature selection and **only to training data** — the test set remains untouched real patient data

### 4. Baseline Models

Three classifiers trained on the ANOVA-selected 50 genes (no SMOTE at baseline):

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight='balanced'` |
| Random Forest | `n_estimators=300`, `class_weight='balanced'` |
| XGBoost | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6` |

Evaluated with **5-fold stratified cross-validation** on the training set, then final evaluation on the hold-out test set.

### 5. Advanced Models

Four advanced approaches trained on the SMOTE-balanced training set:

#### SVM with RBF Kernel
Support Vector Machines with a Radial Basis Function (RBF) kernel are theoretically well-suited to high-dimensional, small-sample problems like gene expression classification. The RBF kernel projects data into a higher-dimensional space where linear separation becomes feasible. A `StandardScaler` is applied within a `Pipeline` to normalize features before the SVM, as SVMs are sensitive to feature scale.
- Configuration: `C=10`, `gamma='scale'`, `class_weight='balanced'`

#### CatBoost
CatBoost is a gradient boosting algorithm developed by Yandex that uses ordered boosting to reduce overfitting. It natively handles class imbalance via `auto_class_weights='Balanced'` and typically requires less hyperparameter tuning than XGBoost.
- Configuration: `iterations=500`, `learning_rate=0.05`, `depth=6`

#### Optuna-Tuned XGBoost (Bayesian Hyperparameter Optimization)
Rather than using default XGBoost parameters, we used **Optuna** — a framework for automated hyperparameter optimization using Tree-structured Parzen Estimators (TPE), a form of Bayesian optimization. Over **50 trials**, Optuna searched the following hyperparameter space:

| Parameter | Search Range |
|---|---|
| `n_estimators` | 100 – 600 |
| `max_depth` | 3 – 10 |
| `learning_rate` | 0.01 – 0.30 (log scale) |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `min_child_weight` | 1 – 10 |

**Best parameters found:**
```
n_estimators: 209
max_depth: 10
learning_rate: 0.01263
subsample: 0.914
colsample_bytree: 0.514
min_child_weight: 1
```

#### Stacking Ensemble
A stacking (stacked generalization) ensemble combines the predictions of SVM, CatBoost, and the Optuna-tuned XGBoost as base learners, feeding their outputs into a **Logistic Regression meta-learner** which learns how to best combine them. This approach exploits the complementary strengths of each model.

---

## Results

### Baseline Model Results

| Model | CV Accuracy (5-Fold) | Test Accuracy |
|---|---|---|
| Logistic Regression | 51.94% ± 7.52% | 64.10% |
| Random Forest | 76.25% ± 7.87% | 82.05% |
| **XGBoost** | **72.38% ± 7.66%** | **84.62%** |

### Advanced Model Results

| Model | CV Accuracy (5-Fold) | Test Accuracy |
|---|---|---|
| SVM (RBF kernel) | 77.00% ± 3.67% | 71.79% |
| CatBoost | 83.00% ± 5.34% | 82.05% |
| XGBoost + Optuna | **87.00%** ± 0.00% | 79.49% |
| Stacking Ensemble | 84.50% ± 5.10% | 79.49% |

### Full Model Comparison

| Model | Test Accuracy | Notes |
|---|---|---|
| Logistic Regression (baseline) | 64.10% | Linear model, struggles with non-linear gene interactions |
| Random Forest (baseline) | 82.05% | Strong ensemble, good out-of-the-box |
| **XGBoost (baseline)** | **84.62%** | **Best on hold-out test** |
| SVM RBF | 71.79% | Strong CV but sensitive to this test split |
| CatBoost | 82.05% | Matches RF; highest CV generalization among baselines |
| XGBoost (Optuna) | 79.49% | Highest CV (87%); slight overfit to training |
| Stacking Ensemble | 79.49% | Best balance; strong CV (84.5%) |

![Model Comparison](model_comparison.png)

### Per-Class Performance — Advanced Models

The classification report below provides precision, recall, and F1-score per class, which is more informative than accuracy for imbalanced medical datasets.

#### SVM (RBF)
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Control | 0.57 | 0.86 | 0.69 | 14 |
| DR | 0.89 | 0.64 | 0.74 | 25 |
| **Accuracy** | | | **0.72** | **39** |

#### CatBoost
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Control | 0.71 | 0.86 | 0.77 | 14 |
| DR | 0.91 | 0.80 | 0.85 | 25 |
| **Accuracy** | | | **0.82** | **39** |

#### XGBoost (Optuna-tuned)
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Control | 0.67 | 0.86 | 0.75 | 14 |
| DR | 0.90 | 0.76 | 0.83 | 25 |
| **Accuracy** | | | **0.79** | **39** |

#### Stacking Ensemble
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Control | 0.69 | 0.79 | 0.73 | 14 |
| DR | 0.87 | 0.80 | 0.83 | 25 |
| **Accuracy** | | | **0.79** | **39** |

> **Clinical note:** In a diagnostic context, **DR recall (sensitivity)** is arguably more important than overall accuracy — a missed DR case (false negative) is more harmful than a false alarm. CatBoost achieves the best DR recall (0.80) among advanced models while maintaining high DR precision (0.91).

---

## Candidate Biomarker Genes

### ANOVA-Selected Top Genes (most statistically discriminative)
These genes showed the greatest mean expression difference between DR and Control groups, ranked by ANOVA F-score:

`NEK6` · `KCNE3` · `RNPEPL1` · `PLXNB2` · `MSRB1` · `Y_RNA` · `DUSP1` · `PRAM1` · `STEAP4` · `EGR1`

### XGBoost Feature Importance — Top 5 Candidate Biomarkers
These genes were identified by the trained XGBoost model as the most influential features for classification. Unlike ANOVA which only measures statistical difference, feature importance captures a gene's actual predictive contribution within a non-linear model:

| Rank | Gene | XGBoost Importance Score | Notes |
|---|---|---|---|
| 1 | **PIEZO1** | 0.0952 | Mechanosensitive ion channel; linked to red blood cell dehydration and vascular integrity |
| 2 | **HIF1A-AS3** | 0.0632 | Antisense RNA for HIF-1α; regulates hypoxia response, highly relevant to retinal ischemia |
| 3 | **PLXNB2** | 0.0488 | Plexin-B2; involved in vascular remodeling and angiogenesis — core DR pathology |
| 4 | **RPS2P5** | 0.0337 | Ribosomal protein pseudogene; emerging roles in cellular stress regulation |
| 5 | **RNF144B** | 0.0337 | E3 ubiquitin ligase; involved in DNA damage response and apoptosis |

![Feature Importances](feature_importance.png)

**Key biological observations:**
- `HIF1A-AS3` regulates the hypoxia-inducible factor pathway, which is directly implicated in the retinal neovascularization seen in advanced DR
- `PLXNB2` appeared in both ANOVA and XGBoost rankings, making it a strong converging candidate
- `PIEZO1` mutations are associated with hereditary xerocytosis and have been linked to vascular complications in diabetes
- `DUSP1` (ANOVA-selected) is a known regulator of the MAPK/ERK signaling pathway, which drives DR-related inflammatory cascades

These genes warrant further validation through qPCR, protein-level studies, and pathway enrichment analysis.

---

## Visualizations

### PCA Plot — DR vs Control
![PCA Plot](pca_plot.png)

Principal Component Analysis was applied to the 50 ANOVA-selected genes (after StandardScaler normalization) to project training samples into 2 dimensions. The degree of cluster separation between DR (red) and Control (blue) in this 2D space reflects how well the selected genes encode disease status. Partial but visible separation confirms the biological signal in the feature set — full separation is not expected with only 2 components from a 50-dimensional space.

### Gene Expression Heatmap — Top 15 Genes
![Heatmap](heatmap.png)

Expression levels of the top 15 ANOVA-ranked genes across all 156 training samples, sorted by class label. Rows are genes; columns are patients. The `RdBu_r` colormap shows relative expression: **red = high expression**, **blue = low expression**. Visible horizontal banding (consistent red or blue across a group of samples) indicates a gene is differentially expressed between DR and Control — exactly the signature expected from a diagnostic biomarker.

### XGBoost Feature Importance — Top 15 Genes
![Feature Importances](feature_importance.png)

Horizontal bar chart of the 15 genes contributing most to XGBoost classification decisions, ranked by the model's internal `feature_importances_` (computed from the average gain across all trees where each feature was used as a split). The top gene (`PIEZO1`) accounts for ~9.5% of total model importance — a notably high share in a 50-feature space.

### Model Comparison — All 7 Classifiers
![Model Comparison](model_comparison.png)

Hold-out test accuracy for all seven classifiers benchmarked in this study. The dashed line marks the baseline XGBoost at 84.62%. Grey bars are baseline models; colored bars are advanced models. Note that CV accuracy (not shown here) favors the advanced models — the gap between CV and test performance reflects variance from the small test set (n=39).

---

## Discussion

### Why the baseline XGBoost held the top spot on the test set

The hold-out test set contains only **39 samples**. At this scale, a single misclassification changes accuracy by 2.56 percentage points — making it highly sensitive to random variance in the specific split. The advanced models actually show superior **cross-validation accuracy** (CatBoost: 83%, Optuna XGBoost: 87%, Stacking: 84.5%), which is a more reliable measure of generalization across the full training distribution.

This is a known problem in small-sample biomedical machine learning studies — test set results should be interpreted alongside CV results, and ideally validated on an independent external cohort.

### On the gap between CV and test accuracy for Optuna XGBoost

The Optuna-tuned model achieved 87% CV accuracy but only 79.49% on the test set. This suggests some degree of **overfitting to the training distribution** during hyperparameter optimization — a common pitfall of extensive tuning on small datasets. Nested cross-validation (using an outer loop for evaluation and inner loop for tuning) would give a less optimistic and more unbiased estimate.

### Class imbalance and clinical relevance

In clinical screening, **sensitivity (recall for DR)** matters more than overall accuracy: a patient with DR who is classified as Control (false negative) may go untreated and lose vision. The best DR recall across all models was **0.86** (SVM RBF and Optuna XGBoost, for Control class), while CatBoost achieved **0.80 DR recall** with **0.91 DR precision** — a favorable balance for screening applications.

### Limitations

1. **Small sample size (n=195):** Limits statistical power and model generalizability. Results should be validated on a larger, independent cohort.
2. **Single train-test split:** A single 80/20 split introduces variance. Repeated k-fold or leave-one-out CV would provide more stable estimates.
3. **No external validation:** All results are on held-out splits of the same dataset. Performance on data from different hospitals or sequencing platforms is unknown.
4. **ANOVA feature selection:** ANOVA tests marginal effects of each gene independently — it does not capture gene-gene interactions. Methods like LASSO or recursive feature elimination with the final model may identify a different and potentially better gene subset.
5. **SMOTE on gene expression:** Synthetic samples generated by SMOTE may not faithfully represent biological gene co-expression patterns, potentially introducing artifacts.

---

## Reproducibility

All results are fully reproducible. The complete pipeline is implemented in `run_analysis.py`.

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost catboost imbalanced-learn optuna matplotlib seaborn

# Place FINAL_COMBINED_DATASET.csv in the project directory, then:
python3 run_analysis.py
```

**Output:**
- Console: all accuracy scores, CV results, classification reports, top biomarker genes
- `pca_plot.png` — PCA visualization
- `heatmap.png` — gene expression heatmap
- `feature_importance.png` — XGBoost feature importances
- `model_comparison.png` — full model comparison chart

| Seed | Value |
|---|---|
| `random_state` | 42 (all sklearn models) |
| `random_seed` | 42 (CatBoost) |
| Optuna trials | 50 |
| CV folds | 5 (StratifiedKFold) |
| Train/test split | 80% / 20% stratified |

---

## Repository Structure

```
Biomarkers_Diabetes/
├── run_analysis.py              # Full pipeline: baseline + advanced models
├── Biomarkers_test_2.ipynb      # Original Colab notebook
├── FINAL_COMBINED_DATASET.csv   # Dataset (gitignored — not pushed to GitHub)
├── pca_plot.png                 # PCA visualization
├── heatmap.png                  # Gene expression heatmap (top 15 genes)
├── feature_importance.png       # XGBoost feature importances (top 15 genes)
├── model_comparison.png         # All 7 models comparison chart
├── .gitignore                   # Excludes dataset and system files
└── README.md                    # This file
```

---

## Requirements

```
pandas >= 1.3
numpy >= 1.21
scikit-learn >= 1.0
xgboost >= 1.6
catboost >= 1.0
imbalanced-learn >= 0.9
optuna >= 3.0
matplotlib >= 3.4
seaborn >= 0.11
```
