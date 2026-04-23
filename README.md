# Biomarker Discovery for Diabetic Retinopathy

A machine learning pipeline for identifying gene expression biomarkers that distinguish Diabetic Retinopathy (DR) patients from healthy controls.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SumitKatuwal3382/Biomarkers_Diabetes/blob/main/Biomarkers_test_2.ipynb)

---

## Overview

Diabetic Retinopathy is a diabetes complication that affects the eyes and is a leading cause of blindness. Early identification through molecular biomarkers could enable better diagnosis and treatment. This project uses gene expression data to build classifiers that distinguish DR patients from controls, and to surface the most discriminative genes as candidate biomarkers.

---

## Dataset

| Property | Value |
|---|---|
| Samples | 195 patients |
| Features | 9,432 gene expression values |
| Classes | DR (125 samples), Control (70 samples) |
| Source | `FINAL_COMBINED_DATASET.csv` |

---

## Pipeline

### 1. Preprocessing
- Labels encoded: `DR = 1`, `Control = 0`
- 80/20 stratified train-test split (156 train / 39 test)

### 2. Feature Selection
- ANOVA F-test (`SelectKBest`) used to rank all 9,432 genes
- Top **50 genes** retained — those with the greatest statistically significant expression differences between DR and Control

Top selected genes include: `NEK6`, `KCNE3`, `RNPEPL1`, `PLXNB2`, `MSRB1`, `DUSP1`, `PRAM1`, `STEAP4`, `EGR1`, and more.

### 3. Model Training & Evaluation

Three classifiers trained and evaluated with 5-fold stratified cross-validation:

| Model | CV Accuracy (5-Fold) | Test Accuracy (Hold-out 20%) |
|---|---|---|
| Logistic Regression | 52.58% ± 7.88% | 64.10% |
| Random Forest | 76.25% ± 7.87% | 82.05% |
| **XGBoost** | **74.27% ± 9.89%** | **84.62%** |

**XGBoost achieved the best test accuracy at 84.62%**, using boosted decision trees that iteratively learn from prior errors — well suited to the non-linear patterns in gene expression data. Logistic regression performed worst, as it is a linear model that struggles with the complex relationships inherent in high-dimensional gene data.

---

## Visualizations

### PCA Plot — DR vs Control
Dimensionality reduction to 2 principal components to visualize class separability in the top-50 gene feature space. Distinct cluster separation validates that the selected genes carry meaningful signal.

### Gene Expression Heatmap
Expression levels of the top 15 ANOVA-ranked genes across all training samples (sorted by class). Red/blue patterns reveal which genes are consistently over- or under-expressed in DR vs Control.

### XGBoost Feature Importances
Bar chart of the top 15 genes ranked by their contribution to the XGBoost model's decisions. These genes are the strongest candidate biomarkers identified by the model.

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```

Install with:

```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```

---

## Usage

The notebook is designed to run on **Google Colab**. Click the badge at the top to open it directly. Upload `FINAL_COMBINED_DATASET.csv` when prompted.

To run locally:

```bash
jupyter notebook Biomarkers_test_2.ipynb
```

---

## Key Findings

- Reducing 9,432 genes down to the top 50 using ANOVA was sufficient for XGBoost to achieve **84.62% accuracy** on unseen samples.
- The PCA plot shows meaningful separation between DR and Control in the selected feature space.
- The heatmap reveals clear differential expression patterns in the top genes.
- XGBoost feature importances identify the most influential candidate biomarkers for further biological investigation.
