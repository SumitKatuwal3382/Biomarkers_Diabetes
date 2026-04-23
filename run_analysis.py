import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["xgboost", "catboost", "lightgbm", "imbalanced-learn", "optuna", "seaborn"]:
    install(pkg)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve,
                              f1_score, precision_score, recall_score)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

DATA_PATH = "FINAL_COMBINED_DATASET.csv"
PLOTS_DIR = "."

# ── Metrics helper ────────────────────────────────────────────────────────────
def compute_metrics(name, y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)          # recall for DR (positive class)
    specificity = tn / (tn + fp)          # recall for Control
    ppv         = tp / (tp + fp)          # precision for DR
    npv         = tn / (tn + fn)          # precision for Control
    fpr         = fp / (fp + tn)          # 1 - specificity
    f1          = f1_score(y_true, y_pred)
    acc         = accuracy_score(y_true, y_pred)
    auc         = roc_auc_score(y_true, y_prob)
    return {
        "Model":       name,
        "Accuracy":    acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV":         ppv,
        "NPV":         npv,
        "FPR":         fpr,
        "F1":          f1,
        "AUC":         auc,
    }

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  LOADING DATA")
print("="*65)
df = pd.read_csv(DATA_PATH)
print(f"  Shape  : {df.shape}")
print(f"  Classes: {df['Label'].value_counts().to_dict()}")

X          = df.drop("Label", axis=1).values
y          = LabelEncoder().fit_transform(df["Label"])   # DR=1, Control=0
gene_names = df.drop("Label", axis=1).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train  : {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Feature selection (ANOVA top 50) ──────────────────────────────────────────
print("\n" + "="*65)
print("  FEATURE SELECTION  —  Top 50 genes (ANOVA F-test)")
print("="*65)
selector    = SelectKBest(f_classif, k=50)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel  = selector.transform(X_test)

selected_genes = [gene_names[i] for i in selector.get_support(indices=True)]
print("  Top 10:", selected_genes[:10])

# ── SMOTE ─────────────────────────────────────────────────────────────────────
X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train_sel, y_train)
print(f"\n  After SMOTE: {X_train_sm.shape}  |  "
      f"Classes: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ═════════════════════════════════════════════════════════════════════════════
# MODELS
# ═════════════════════════════════════════════════════════════════════════════

all_metrics   = []   # list of metric dicts
roc_data      = {}   # name → (fpr_arr, tpr_arr, auc)

def evaluate(name, model, X_tr, y_tr, X_te, y_te, needs_scale=False):
    """Fit, predict, collect all metrics and ROC data."""
    if needs_scale:
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_prob = model.decision_function(X_te)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    m = compute_metrics(name, y_te, y_pred, y_prob)
    m["CV_mean"] = cv_scores.mean()
    m["CV_std"]  = cv_scores.std()
    all_metrics.append(m)

    fpr_arr, tpr_arr, _ = roc_curve(y_te, y_prob)
    roc_data[name] = (fpr_arr, tpr_arr, m["AUC"])

    print(f"\n  [{name}]")
    print(f"    CV Acc : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test   : Acc={m['Accuracy']:.4f}  AUC={m['AUC']:.4f}  "
          f"Sens={m['Sensitivity']:.4f}  Spec={m['Specificity']:.4f}")
    print(f"             PPV={m['PPV']:.4f}  NPV={m['NPV']:.4f}  "
          f"FPR={m['FPR']:.4f}  F1={m['F1']:.4f}")
    return model

# ── Baseline models (no SMOTE) ────────────────────────────────────────────────
print("\n" + "="*65)
print("  BASELINE MODELS  (original train set, no SMOTE)")
print("="*65)

evaluate("Logistic Regression",
         LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
         X_train_sel, y_train, X_test_sel, y_test)

evaluate("Random Forest",
         RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42, n_jobs=-1),
         X_train_sel, y_train, X_test_sel, y_test)

evaluate("XGBoost (baseline)",
         XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                       eval_metric="logloss", random_state=42),
         X_train_sel, y_train, X_test_sel, y_test)

# ── Advanced models (SMOTE-balanced) ─────────────────────────────────────────
print("\n" + "="*65)
print("  ADVANCED MODELS  (SMOTE-balanced train set)")
print("="*65)

evaluate("AdaBoost",
         AdaBoostClassifier(
             estimator=DecisionTreeClassifier(max_depth=2),
             n_estimators=200, learning_rate=0.5, random_state=42),
         X_train_sm, y_train_sm, X_test_sel, y_test)

evaluate("SVM (RBF)",
         SVC(kernel="rbf", class_weight="balanced", C=10,
             gamma="scale", probability=True, random_state=42),
         X_train_sm, y_train_sm, X_test_sel, y_test, needs_scale=True)

evaluate("LightGBM",
         LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                        class_weight="balanced", random_state=42, verbose=-1),
         X_train_sm, y_train_sm, X_test_sel, y_test)

evaluate("CatBoost",
         CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                            auto_class_weights="Balanced", random_seed=42, verbose=0),
         X_train_sm, y_train_sm, X_test_sel, y_test)

# ── Optuna XGBoost ────────────────────────────────────────────────────────────
print("\n  [Optuna XGBoost — 50 trials Bayesian search]")

def xgb_objective(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 600),
        max_depth        = trial.suggest_int("max_depth", 3, 10),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        eval_metric="logloss", random_state=42,
    )
    return cross_val_score(XGBClassifier(**p), X_train_sm, y_train_sm,
                           cv=cv, scoring="accuracy").mean()

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=50)
print(f"    Best CV: {xgb_study.best_value:.4f}  |  {xgb_study.best_params}")

evaluate("XGBoost (Optuna)",
         XGBClassifier(**xgb_study.best_params, eval_metric="logloss", random_state=42),
         X_train_sm, y_train_sm, X_test_sel, y_test)

# ── Optuna LightGBM ───────────────────────────────────────────────────────────
print("\n  [Optuna LightGBM — 50 trials Bayesian search]")

def lgbm_objective(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 600),
        max_depth        = trial.suggest_int("max_depth", 3, 10),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        num_leaves       = trial.suggest_int("num_leaves", 20, 150),
        min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
        class_weight="balanced", random_state=42, verbose=-1,
    )
    return cross_val_score(LGBMClassifier(**p), X_train_sm, y_train_sm,
                           cv=cv, scoring="accuracy").mean()

lgbm_study = optuna.create_study(direction="maximize")
lgbm_study.optimize(lgbm_objective, n_trials=50)
print(f"    Best CV: {lgbm_study.best_value:.4f}  |  {lgbm_study.best_params}")

evaluate("LightGBM (Optuna)",
         LGBMClassifier(**lgbm_study.best_params, class_weight="balanced",
                        random_state=42, verbose=-1),
         X_train_sm, y_train_sm, X_test_sel, y_test)

# ── Stacking Ensemble ─────────────────────────────────────────────────────────
print("\n  [Stacking Ensemble — SVM + LightGBM(Optuna) + XGBoost(Optuna) → LR]")

sc_stack = StandardScaler()
X_tr_stack = sc_stack.fit_transform(X_train_sm)
X_te_stack = sc_stack.transform(X_test_sel)

estimators = [
    ("svm",   SVC(kernel="rbf", class_weight="balanced", C=10,
                  gamma="scale", probability=True, random_state=42)),
    ("lgbm",  LGBMClassifier(**lgbm_study.best_params, class_weight="balanced",
                              random_state=42, verbose=-1)),
    ("xgb",   XGBClassifier(**xgb_study.best_params, eval_metric="logloss",
                             random_state=42)),
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
    cv=5,
)
evaluate("Stacking Ensemble",
         stack, X_tr_stack, y_train_sm, X_te_stack, y_test)

# ═════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. ROC curves ─────────────────────────────────────────────────────────────
print("\n  Generating plots...")
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))

for (name, (fpr_arr, tpr_arr, auc_val)), color in zip(roc_data.items(), colors):
    ax.plot(fpr_arr, tpr_arr, lw=2, color=color,
            label=f"{name}  (AUC = {auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()
print("  [Saved] roc_curves.png")

# ── 2. Metrics heatmap ────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics).set_index("Model")
plot_cols   = ["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "AUC"]
hm_data     = metrics_df[plot_cols].astype(float)

fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(hm_data, annot=True, fmt=".3f", cmap="YlGn",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Score"},
            vmin=0.5, vmax=1.0)
ax.set_title("Performance Metrics — All Models", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("metrics_heatmap.png", dpi=150)
plt.close()
print("  [Saved] metrics_heatmap.png")

# ── 3. Accuracy + AUC bar chart ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
model_names = metrics_df.index.tolist()
x = np.arange(len(model_names))
bar_colors  = (["#90A4AE"] * 3) + (["#42A5F5", "#66BB6A", "#FFA726",
                                      "#AB47BC", "#EF5350", "#26C6DA", "#FF7043"])
bar_colors  = bar_colors[:len(model_names)]

for ax, metric, title in zip(axes,
                              ["Accuracy", "AUC"],
                              ["Test Accuracy", "AUC-ROC"]):
    vals = metrics_df[metric].values
    bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

plt.suptitle("Model Comparison — Test Accuracy & AUC", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()
print("  [Saved] model_comparison.png")

# ── 4. PCA plot ───────────────────────────────────────────────────────────────
sc_pca  = StandardScaler()
X_sc    = sc_pca.fit_transform(X_train_sel)
pca     = PCA(n_components=2)
X_pca   = pca.fit_transform(X_sc)

fig, ax = plt.subplots(figsize=(8, 6))
for i, (color, label) in enumerate(zip(["#2196F3", "#F44336"], ["Control", "DR"])):
    mask = y_train == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label,
               alpha=0.7, edgecolors="white", s=60)
ax.set_title("PCA — DR vs Control (top 50 genes)", fontsize=14, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pca_plot.png", dpi=150)
plt.close()
print("  [Saved] pca_plot.png")

# ── 5. Gene expression heatmap ────────────────────────────────────────────────
top15_idx   = selector.scores_[selector.get_support(indices=True)].argsort()[::-1][:15]
top15_genes = [selected_genes[i] for i in top15_idx]
heatmap_df  = pd.DataFrame(X_train_sel[:, top15_idx], columns=top15_genes)
heatmap_df["Label"] = ["DR" if v == 1 else "Control" for v in y_train]
heatmap_df  = heatmap_df.sort_values("Label")

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(heatmap_df[top15_genes].T, cmap="RdBu_r", xticklabels=False,
            yticklabels=True, center=0, cbar_kws={"label": "Expression Level"}, ax=ax)
ax.set_title("Gene Expression Heatmap — Top 15 ANOVA Genes", fontsize=14, fontweight="bold")
ax.set_xlabel("Samples (sorted by class)"); ax.set_ylabel("Genes")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.close()
print("  [Saved] heatmap.png")

# ── 6. XGBoost feature importance ────────────────────────────────────────────
best_xgb_model = [m for m in all_metrics if m["Model"] == "XGBoost (Optuna)"]
xgb_eval = XGBClassifier(**xgb_study.best_params, eval_metric="logloss", random_state=42)
xgb_eval.fit(X_train_sm, y_train_sm)
imp       = xgb_eval.feature_importances_
top15i    = np.argsort(imp)[::-1][:15]
top15g    = [selected_genes[i] for i in top15i]
top15s    = imp[top15i]

fig, ax = plt.subplots(figsize=(10, 6))
bar_imp_colors = ["#4CAF50" if i == 0 else "#2196F3" for i in range(15)]
ax.barh(top15g[::-1], top15s[::-1], color=bar_imp_colors[::-1], edgecolor="white")
ax.set_title("Top 15 Feature Importances — XGBoost (Optuna)", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score"); ax.set_ylabel("Gene")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("  [Saved] feature_importance.png")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*95)
print("  FINAL RESULTS SUMMARY")
print("="*95)

summary = metrics_df[["CV_mean", "CV_std", "Accuracy", "Sensitivity",
                        "Specificity", "PPV", "NPV", "FPR", "F1", "AUC"]]
summary.columns = ["CV Acc", "CV Std", "Acc", "Sens", "Spec", "PPV", "NPV", "FPR", "F1", "AUC"]

header = f"{'Model':<28}" + "".join(f"{c:>8}" for c in summary.columns)
print(header)
print("-" * len(header))

best_acc = summary["Acc"].max()
best_auc = summary["AUC"].max()

for model_name, row in summary.iterrows():
    flags = ""
    if row["Acc"] == best_acc: flags += " ← best acc"
    if row["AUC"] == best_auc: flags += " ← best AUC"
    vals = "".join(f"{v:>8.3f}" for v in row.values)
    print(f"{model_name:<28}{vals}{flags}")

print("\n  Top 5 biomarker genes (XGBoost Optuna feature importance):")
for g, s in zip(top15g[:5], top15s[:5]):
    print(f"    {g:<18} {s:.4f}")

print("\n  Plots saved:")
for p in ["roc_curves.png", "metrics_heatmap.png", "model_comparison.png",
          "pca_plot.png", "heatmap.png", "feature_importance.png"]:
    print(f"    {p}")
