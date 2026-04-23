import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["xgboost", "catboost", "imbalanced-learn", "optuna", "seaborn"]:
    install(pkg)

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
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

DATA_PATH = "FINAL_COMBINED_DATASET.csv"

# ── Load data ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Classes: {df['Label'].value_counts().to_dict()}")

# ── Encode + split ───────────────────────────────────────────────────────────
X = df.drop("Label", axis=1).values
y = LabelEncoder().fit_transform(df["Label"])   # DR=1, Control=0
gene_names = df.drop("Label", axis=1).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Feature selection (top 50 genes via ANOVA) ───────────────────────────────
print("\n" + "="*60)
print("FEATURE SELECTION — Top 50 genes (ANOVA)")
print("="*60)
selector = SelectKBest(f_classif, k=50)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel  = selector.transform(X_test)

selected_genes = [gene_names[i] for i in selector.get_support(indices=True)]
print("Top 10 selected genes:", selected_genes[:10])

# ── Baseline models ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE MODELS")
print("="*60)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baselines = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                         eval_metric="logloss", random_state=42),
}

baseline_results = {}
print(f"\n{'Model':<25}  {'CV Acc':>10}  {'Test Acc':>10}")
print("-" * 50)
for name, model in baselines.items():
    cv_scores = cross_val_score(model, X_train_sel, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train_sel, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test_sel))
    baseline_results[name] = test_acc
    print(f"{name:<25}  {cv_scores.mean():.4f}±{cv_scores.std():.4f}  {test_acc:.4f} ({test_acc*100:.2f}%)")

# ── Visualisation 1: PCA ─────────────────────────────────────────────────────
scaler_pca = StandardScaler()
X_pca = PCA(n_components=2).fit_transform(scaler_pca.fit_transform(X_train_sel))
pca_obj = PCA(n_components=2).fit(scaler_pca.transform(X_train_sel))

plt.figure(figsize=(8, 6))
for i, (color, label) in enumerate(zip(["#2196F3", "#F44336"], ["Control", "DR"])):
    mask = y_train == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label,
                alpha=0.7, edgecolors="white", s=60)
plt.title("PCA Plot — DR vs Control", fontsize=14, fontweight="bold")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("pca_plot.png", dpi=150)
plt.close()
print("\n[Saved] pca_plot.png")

# ── Visualisation 2: Heatmap ─────────────────────────────────────────────────
top15_idx   = selector.scores_[selector.get_support(indices=True)].argsort()[::-1][:15]
top15_genes = [selected_genes[i] for i in top15_idx]
heatmap_df  = pd.DataFrame(X_train_sel[:, top15_idx], columns=top15_genes)
heatmap_df["Label"] = ["DR" if v == 1 else "Control" for v in y_train]
heatmap_df = heatmap_df.sort_values("Label")

plt.figure(figsize=(14, 7))
sns.heatmap(heatmap_df[top15_genes].T, cmap="RdBu_r", xticklabels=False,
            yticklabels=True, center=0, cbar_kws={"label": "Expression Level"})
plt.title("Gene Expression Heatmap — Top 15 Genes", fontsize=14, fontweight="bold")
plt.xlabel("Samples (sorted by class)"); plt.ylabel("Genes")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.close()
print("[Saved] heatmap.png")

# ── Visualisation 3: XGBoost feature importance ──────────────────────────────
xgb_base = baselines["XGBoost"]
imp    = xgb_base.feature_importances_
top15i = np.argsort(imp)[::-1][:15]
top15g = [selected_genes[i] for i in top15i]
top15s = imp[top15i]

plt.figure(figsize=(10, 6))
colors_imp = ["#4CAF50" if i == 0 else "#2196F3" for i in range(15)]
plt.barh(top15g[::-1], top15s[::-1], color=colors_imp[::-1], edgecolor="white")
plt.title("Top 15 Feature Importances — XGBoost", fontsize=14, fontweight="bold")
plt.xlabel("Importance Score"); plt.ylabel("Gene")
plt.grid(axis="x", alpha=0.3); plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("[Saved] feature_importance.png")
print("\nTop 5 biomarker genes:")
for g, s in zip(top15g[:5], top15s[:5]):
    print(f"  {g:<15} {s:.4f}")

# ── SMOTE ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ADVANCED MODELS")
print("="*60)
X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train_sel, y_train)
print(f"\nAfter SMOTE — Train shape: {X_train_sm.shape}")
print(f"Class distribution: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

advanced_results = {}

# ── SVM ──────────────────────────────────────────────────────────────────────
print("\n--- SVM (RBF) ---")
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", class_weight="balanced", C=10, gamma="scale", random_state=42)),
])
svm_cv = cross_val_score(svm_pipe, X_train_sm, y_train_sm, cv=cv, scoring="accuracy")
svm_pipe.fit(X_train_sm, y_train_sm)
svm_pred = svm_pipe.predict(X_test_sel)
svm_acc  = accuracy_score(y_test, svm_pred)
advanced_results["SVM (RBF)"] = svm_acc
print(f"CV:   {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")
print(f"Test: {svm_acc:.4f} ({svm_acc*100:.2f}%)")
print(classification_report(y_test, svm_pred, target_names=["Control", "DR"]))

# ── CatBoost ─────────────────────────────────────────────────────────────────
print("--- CatBoost ---")
cat = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                         auto_class_weights="Balanced", random_seed=42, verbose=0)
cat_cv = cross_val_score(cat, X_train_sm, y_train_sm, cv=cv, scoring="accuracy")
cat.fit(X_train_sm, y_train_sm)
cat_pred = cat.predict(X_test_sel)
cat_acc  = accuracy_score(y_test, cat_pred)
advanced_results["CatBoost"] = cat_acc
print(f"CV:   {cat_cv.mean():.4f} ± {cat_cv.std():.4f}")
print(f"Test: {cat_acc:.4f} ({cat_acc*100:.2f}%)")
print(classification_report(y_test, cat_pred, target_names=["Control", "DR"]))

# ── Optuna XGBoost ───────────────────────────────────────────────────────────
print("--- XGBoost + Optuna (50 trials) ---")

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eval_metric": "logloss", "random_state": 42,
    }
    return cross_val_score(XGBClassifier(**params), X_train_sm, y_train_sm,
                           cv=cv, scoring="accuracy").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(f"Best CV: {study.best_value:.4f}  |  Params: {study.best_params}")

xgb_tuned = XGBClassifier(**study.best_params, eval_metric="logloss", random_state=42)
xgb_tuned.fit(X_train_sm, y_train_sm)
xgb_tuned_pred = xgb_tuned.predict(X_test_sel)
xgb_tuned_acc  = accuracy_score(y_test, xgb_tuned_pred)
advanced_results["XGBoost (Optuna)"] = xgb_tuned_acc
print(f"Test: {xgb_tuned_acc:.4f} ({xgb_tuned_acc*100:.2f}%)")
print(classification_report(y_test, xgb_tuned_pred, target_names=["Control", "DR"]))

# ── Stacking Ensemble ────────────────────────────────────────────────────────
print("--- Stacking Ensemble ---")
estimators = [
    ("svm", Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", C=10, gamma="scale",
                    probability=True, random_state=42)),
    ])),
    ("catboost", CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                    auto_class_weights="Balanced", random_seed=42, verbose=0)),
    ("xgb_tuned", XGBClassifier(**study.best_params, eval_metric="logloss", random_state=42)),
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
    cv=5,
)
stack_cv = cross_val_score(stack, X_train_sm, y_train_sm, cv=cv, scoring="accuracy")
stack.fit(X_train_sm, y_train_sm)
stack_pred = stack.predict(X_test_sel)
stack_acc  = accuracy_score(y_test, stack_pred)
advanced_results["Stacking Ensemble"] = stack_acc
print(f"CV:   {stack_cv.mean():.4f} ± {stack_cv.std():.4f}")
print(f"Test: {stack_acc:.4f} ({stack_acc*100:.2f}%)")
print(classification_report(y_test, stack_pred, target_names=["Control", "DR"]))

# ── Final comparison chart ───────────────────────────────────────────────────
all_results = {
    "Logistic Regression (baseline)": baseline_results["Logistic Regression"],
    "Random Forest (baseline)":       baseline_results["Random Forest"],
    "XGBoost (baseline)":             baseline_results["XGBoost"],
    **{f"{k} ★": v for k, v in advanced_results.items()},
}

names  = list(all_results.keys())
accs   = [v * 100 for v in all_results.values()]
colors = ["#90A4AE"] * 3 + ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350"]

plt.figure(figsize=(13, 7))
bars = plt.barh(names, accs, color=colors, edgecolor="white")
plt.axvline(84.62, color="gray", linestyle="--", linewidth=1.2, label="XGBoost baseline (84.62%)")
for bar, acc in zip(bars, accs):
    plt.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
             f"{acc:.2f}%", va="center", fontsize=10)
plt.xlim(50, 108)
plt.xlabel("Test Accuracy (%)")
plt.title("Model Comparison — Baseline vs Advanced", fontsize=14, fontweight="bold")
plt.legend(); plt.grid(axis="x", alpha=0.3); plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()
print("\n[Saved] model_comparison.png")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"\n{'Model':<35}  {'Test Acc':>10}")
print("-" * 48)
for name, acc in all_results.items():
    marker = " <-- best" if acc == max(all_results.values()) else ""
    print(f"{name:<35}  {acc*100:>8.2f}%{marker}")

print("\nPlots saved: pca_plot.png | heatmap.png | feature_importance.png | model_comparison.png")
