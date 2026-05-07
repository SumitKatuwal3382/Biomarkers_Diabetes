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
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE


def compute_metrics(name, y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return dict(
        Model       = name,
        Accuracy    = accuracy_score(y_true, y_pred),
        Sensitivity = tp / (tp + fn),
        Specificity = tn / (tn + fp),
        PPV         = tp / (tp + fp),
        NPV         = tn / (tn + fn),
        FPR         = fp / (fp + tn),
        F1          = f1_score(y_true, y_pred),
        AUC         = roc_auc_score(y_true, y_prob),
    )


# --- Data loading ---

df         = pd.read_csv("FINAL_COMBINED_DATASET.csv")
X          = df.drop("Label", axis=1).values.astype(float)
y          = LabelEncoder().fit_transform(df["Label"])
gene_names = df.drop("Label", axis=1).columns.tolist()

print(f"Dataset: {df.shape[0]} samples, {df.shape[1]-1} genes")
print(f"Classes: {df['Label'].value_counts().to_dict()}")


# --- Preprocessing ---

sample_means = X.mean(axis=1)
dr_mean      = X[y == 1].mean()
ctl_mean     = X[y == 0].mean()

print(f"\nGlobal expression — DR: {dr_mean:.4f}  Control: {ctl_mean:.4f}  delta: {dr_mean - ctl_mean:.4f}")

X_centered        = X - sample_means[:, None]
gene_var_centered = X_centered.var(axis=0)
gene_var_raw      = X.var(axis=0)
gene_mean_raw     = X.mean(axis=0)

print("\nTop 10 highly variable genes (centered variance):")
for i in gene_var_centered.argsort()[::-1][:10]:
    print(f"  {gene_names[i]:<22}  var={gene_var_centered[i]:.3f}  mean={gene_mean_raw[i]:.3f}")


# --- Train / test split ---

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]}  Test: {X_test.shape[0]}")


# --- Feature selection: ANOVA F-test, k=50 ---

selector    = SelectKBest(f_classif, k=50)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel  = selector.transform(X_test)

selected_genes = [gene_names[i] for i in selector.get_support(indices=True)]
top10_idx      = selector.scores_[selector.get_support()].argsort()[::-1][:10]

print("\nTop 10 ANOVA-selected genes:")
for rank, idx in enumerate(top10_idx, 1):
    print(f"  {rank:2d}. {selected_genes[idx]}")


# --- SMOTE oversampling ---

X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train_sel, y_train)
print(f"\nAfter SMOTE: {X_train_sm.shape}  classes: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# --- Model evaluation ---

all_metrics = []
roc_data    = {}


def evaluate(name, model, X_tr, y_tr, X_te, y_te, needs_scale=False):
    if needs_scale:
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = (model.predict_proba(X_te)[:, 1]
              if hasattr(model, "predict_proba")
              else (lambda d: (d - d.min()) / (d.max() - d.min()))(model.decision_function(X_te)))
    m = compute_metrics(name, y_te, y_pred, y_prob)
    m["CV_mean"] = cv_scores.mean()
    m["CV_std"]  = cv_scores.std()
    all_metrics.append(m)
    fpr_arr, tpr_arr, _ = roc_curve(y_te, y_prob)
    roc_data[name] = (fpr_arr, tpr_arr, m["AUC"])
    print(f"\n{name}")
    print(f"  CV : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Acc={m['Accuracy']:.4f}  AUC={m['AUC']:.4f}  "
          f"Sens={m['Sensitivity']:.4f}  Spec={m['Specificity']:.4f}  "
          f"PPV={m['PPV']:.4f}  NPV={m['NPV']:.4f}  F1={m['F1']:.4f}")
    return model


print("\n--- Baseline models (no SMOTE) ---")
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


print("\n--- SMOTE-balanced models ---")
evaluate("AdaBoost",
         AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2),
                            n_estimators=200, learning_rate=0.5, random_state=42),
         X_train_sm, y_train_sm, X_test_sel, y_test)

evaluate("SVM (RBF)",
         SVC(kernel="rbf", class_weight="balanced", C=10, gamma="scale",
             probability=True, random_state=42),
         X_train_sm, y_train_sm, X_test_sel, y_test, needs_scale=True)

evaluate("LightGBM",
         LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                        class_weight="balanced", random_state=42, verbose=-1),
         X_train_sm, y_train_sm, X_test_sel, y_test)

evaluate("CatBoost",
         CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                            auto_class_weights="Balanced", random_seed=42, verbose=0),
         X_train_sm, y_train_sm, X_test_sel, y_test)


print("\n--- Optuna XGBoost (75 trials) ---")

def xgb_obj(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 800),
        max_depth         = trial.suggest_int("max_depth", 3, 10),
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
        gamma             = trial.suggest_float("gamma", 0, 5),
        reg_alpha         = trial.suggest_float("reg_alpha", 0, 2),
        reg_lambda        = trial.suggest_float("reg_lambda", 0, 2),
        eval_metric       = "logloss",
        random_state      = 42,
    )
    return cross_val_score(XGBClassifier(**params), X_train_sm, y_train_sm,
                           cv=cv, scoring="accuracy").mean()

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_obj, n_trials=75)
print(f"Best CV: {xgb_study.best_value:.4f}")

evaluate("XGBoost (Optuna)",
         XGBClassifier(**xgb_study.best_params, eval_metric="logloss", random_state=42),
         X_train_sm, y_train_sm, X_test_sel, y_test)


print("\n--- Optuna LightGBM (75 trials) ---")

def lgbm_obj(trial):
    params = dict(
        n_estimators       = trial.suggest_int("n_estimators", 100, 800),
        max_depth          = trial.suggest_int("max_depth", 3, 10),
        learning_rate      = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        subsample          = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree   = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        num_leaves         = trial.suggest_int("num_leaves", 20, 150),
        min_child_samples  = trial.suggest_int("min_child_samples", 5, 50),
        class_weight       = "balanced",
        random_state       = 42,
        verbose            = -1,
    )
    return cross_val_score(LGBMClassifier(**params), X_train_sm, y_train_sm,
                           cv=cv, scoring="accuracy").mean()

lgbm_study = optuna.create_study(direction="maximize")
lgbm_study.optimize(lgbm_obj, n_trials=75)
print(f"Best CV: {lgbm_study.best_value:.4f}")

evaluate("LightGBM (Optuna)",
         LGBMClassifier(**lgbm_study.best_params, class_weight="balanced",
                        random_state=42, verbose=-1),
         X_train_sm, y_train_sm, X_test_sel, y_test)


print("\n--- Stacking Ensemble (SVM + LightGBM + XGBoost -> LR) ---")

sc_stack   = StandardScaler()
X_tr_stack = sc_stack.fit_transform(X_train_sm)
X_te_stack = sc_stack.transform(X_test_sel)

stack = StackingClassifier(
    estimators=[
        ("svm",  SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                     class_weight="balanced", random_state=42)),
        ("lgbm", LGBMClassifier(**lgbm_study.best_params, class_weight="balanced",
                                random_state=42, verbose=-1)),
        ("xgb",  XGBClassifier(**xgb_study.best_params, eval_metric="logloss",
                               random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
    cv=5,
)
evaluate("Stacking Ensemble", stack, X_tr_stack, y_train_sm, X_te_stack, y_test)


# --- Figures ---

# Figure 1: Preprocessing overview
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(X.flatten()[::50], bins=80, color="#2196F3", alpha=0.8, edgecolor="none")
axes[0].axvline(X.mean(), color="#F44336", lw=2, label=f"Mean = {X.mean():.1f}")
axes[0].set_title("Expression Value Distribution", fontsize=10, fontweight="bold")
axes[0].set_xlabel("Expression value")
axes[0].set_ylabel("Frequency")
axes[0].legend(fontsize=9)

top50_mask = np.zeros(len(gene_var_raw), dtype=bool)
top50_mask[selector.get_support(indices=True)] = True
axes[1].scatter(gene_mean_raw, gene_var_raw, alpha=0.15, s=4, color="#90A4AE")
axes[1].scatter(gene_mean_raw[top50_mask], gene_var_raw[top50_mask],
                alpha=0.9, s=30, color="#F44336", zorder=3, label="ANOVA top 50")
axes[1].set_xlabel("Gene mean expression")
axes[1].set_ylabel("Gene variance")
axes[1].set_title("Mean–Variance Relationship", fontsize=10, fontweight="bold")
axes[1].legend(fontsize=9)

axes[2].hist(sample_means[y == 0], bins=20, alpha=0.7, color="#2196F3",
             label="Control", edgecolor="none")
axes[2].hist(sample_means[y == 1], bins=20, alpha=0.7, color="#F44336",
             label="DR", edgecolor="none")
axes[2].axvline(ctl_mean, color="#1565C0", lw=2, ls="--", label=f"Control mean = {ctl_mean:.2f}")
axes[2].axvline(dr_mean,  color="#B71C1C", lw=2, ls="--", label=f"DR mean = {dr_mean:.2f}")
axes[2].set_xlabel("Per-sample mean expression")
axes[2].set_ylabel("Count")
axes[2].set_title(f"Global Expression by Class  (delta = {dr_mean - ctl_mean:.2f})",
                  fontsize=10, fontweight="bold")
axes[2].legend(fontsize=8)

plt.suptitle("Preprocessing Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("preprocessing_diagnosis.png", dpi=150)
plt.close()


# Figure 2: LDA class separation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors_cls = {0: "#2196F3", 1: "#F44336"}
labels_cls = {0: "Control",  1: "DR"}

X_scaled = StandardScaler().fit_transform(X_train_sel)
lda      = LinearDiscriminantAnalysis()
ld1      = lda.fit_transform(X_scaled, y_train)[:, 0]

for cls, label, color in [(0, "Control", "#2196F3"), (1, "DR", "#F44336")]:
    vals = ld1[y_train == cls]
    axes[0].hist(vals, bins=20, alpha=0.45, color=color, label=label,
                 edgecolor="none", density=True)
    sns.kdeplot(vals, ax=axes[0], color=color, lw=2.5)
axes[0].set_title("LD1 Score Distribution", fontsize=11, fontweight="bold")
axes[0].set_xlabel("LD1 Score")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].grid(alpha=0.3)

rng    = np.random.default_rng(42)
jitter = rng.uniform(-0.3, 0.3, len(ld1))
for cls in [0, 1]:
    mask = y_train == cls
    axes[1].scatter(ld1[mask], jitter[mask],
                    c=colors_cls[cls], label=labels_cls[cls],
                    alpha=0.7, s=45, edgecolors="white", lw=0.4)
axes[1].set_title("Sample Distribution along LD1", fontsize=11, fontweight="bold")
axes[1].set_xlabel("LD1 Score")
axes[1].set_yticks([])
axes[1].legend()
axes[1].grid(axis="x", alpha=0.3)

plt.suptitle("Linear Discriminant Analysis — Control vs DR", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("lda_plot.png", dpi=150)
plt.close()


# Figure 3: Gene expression heatmap
top15_idx   = selector.scores_[selector.get_support()].argsort()[::-1][:15]
top15_genes = [selected_genes[i] for i in top15_idx]
hm          = pd.DataFrame(X_train_sel[:, top15_idx], columns=top15_genes)
hm["Label"] = ["DR" if v == 1 else "Control" for v in y_train]
hm          = hm.sort_values("Label")

hm_data   = hm[top15_genes]
hm_zscore = (hm_data - hm_data.mean()) / hm_data.std()

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(hm_zscore.T, cmap="RdBu_r", xticklabels=False, yticklabels=True,
            center=0, cbar_kws={"label": "z-score"}, ax=ax)
ax.set_title("Gene Expression Heatmap — Top 15 ANOVA Genes",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Samples (sorted by class)")
ax.set_ylabel("Genes")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.close()


# Figure 4: ROC curves
fig, ax = plt.subplots(figsize=(10, 8))
for (name, (fpr_arr, tpr_arr, auc_val)), color in zip(
        roc_data.items(), plt.cm.tab10(np.linspace(0, 1, len(roc_data)))):
    ax.plot(fpr_arr, tpr_arr, lw=2, color=color, label=f"{name}  (AUC = {auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()


# Figure 5: Metrics heatmap
metrics_df = pd.DataFrame(all_metrics).set_index("Model")
plot_cols  = ["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "AUC"]

fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(metrics_df[plot_cols].astype(float), annot=True, fmt=".3f",
            cmap="YlGn", linewidths=0.5, ax=ax,
            cbar_kws={"label": "Score"}, vmin=0.5, vmax=1.0)
ax.set_title("Performance Metrics — All Models", fontsize=14, fontweight="bold")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("metrics_heatmap.png", dpi=150)
plt.close()


# Figure 6: Accuracy and AUC bar charts
fig, axes2 = plt.subplots(1, 2, figsize=(16, 6))
model_names = metrics_df.index.tolist()
x           = np.arange(len(model_names))
bar_colors  = (["#90A4AE"] * 3 + ["#42A5F5", "#66BB6A", "#FFA726",
                                    "#AB47BC", "#EF5350", "#26C6DA", "#FF7043"])[:len(model_names)]

for ax2, metric, title in zip(axes2, ["Accuracy", "AUC"], ["Test Accuracy", "AUC-ROC"]):
    vals = metrics_df[metric].values
    bars = ax2.bar(x, vals, color=bar_colors, edgecolor="white", width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=35, ha="right", fontsize=9)
    ax2.set_ylim(0.5, 1.08)
    ax2.set_title(title, fontsize=13, fontweight="bold")
    ax2.set_ylabel("Score")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

plt.suptitle("Model Comparison — Test Accuracy & AUC", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()


# Figure 7: Feature importances
xgb_fi = XGBClassifier(**xgb_study.best_params, eval_metric="logloss", random_state=42)
xgb_fi.fit(X_train_sm, y_train_sm)
imp    = xgb_fi.feature_importances_
top15i = np.argsort(imp)[::-1][:15]
top15g = [selected_genes[i] for i in top15i]
top15s = imp[top15i]

fig, ax = plt.subplots(figsize=(10, 6))
imp_colors = ["#4CAF50" if i == 0 else "#2196F3" for i in range(15)]
ax.barh(top15g[::-1], top15s[::-1], color=imp_colors[::-1], edgecolor="white")
ax.set_title("Top 15 Feature Importances — XGBoost (Optuna)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Gene")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()


# --- Results summary ---

summary = metrics_df[["CV_mean", "CV_std", "Accuracy", "Sensitivity",
                       "Specificity", "PPV", "NPV", "FPR", "F1", "AUC"]]
summary.columns = ["CV", "+/-", "Acc", "Sens", "Spec", "PPV", "NPV", "FPR", "F1", "AUC"]
best_acc = summary["Acc"].max()
best_auc = summary["AUC"].max()

print("\n" + "-" * 100)
hdr = f"{'Model':<28}" + "".join(f"{c:>7}" for c in summary.columns)
print(hdr)
print("-" * len(hdr))
for mname, row in summary.iterrows():
    flags = ""
    if row["Acc"] == best_acc:
        flags += "  <- best acc"
    if row["AUC"] == best_auc:
        flags += "  <- best AUC"
    print(f"{mname:<28}" + "".join(f"{v:>7.3f}" for v in row.values) + flags)

print("\nTop 5 candidate biomarkers (XGBoost Optuna):")
for gene, score in zip(top15g[:5], top15s[:5]):
    print(f"  {gene:<22}  {score:.4f}")
