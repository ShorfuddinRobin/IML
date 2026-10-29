
def _ensure(pkgs):
    import importlib, subprocess, sys
    to_install = []
    for p in pkgs:
        try:
            importlib.import_module(p if p != "sklearn" else "sklearn")
        except ImportError:
            to_install.append(p)
    if to_install:
        print("Installing:", to_install)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])

_ensure(["numpy", "pandas", "matplotlib", "scikit-learn", "shap", "lime"])

# ---- 1) Imports ----
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import PartialDependenceDisplay

import shap
from lime.lime_tabular import LimeTabularExplainer

# ---- 2) Output config ----
OUT = "slides_figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.figsize"] = (6, 4)

# ---- 3) Load data ----
# Replace this with a local path if offline:
FILE_ID = "1O_q0g7vO0BrVXRd2tyibaBK-R9TFzeKM"   # Kaggle Stroke CSV on Drive
url = f"https://drive.google.com/uc?id={FILE_ID}"

try:
    df = pd.read_csv(url)
except Exception as e:
    raise RuntimeError(
        "Failed to load CSV from Google Drive. If you're offline, replace the URL with a local file path, e.g., pd.read_csv('stroke.csv')."
    ) from e

# ---- 4) Preprocess ----
# Handle missing BMI
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

# Encode categorical columns
cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
le_map = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))
    le_map[c] = le

# Features/target
X = df.drop(columns=["id", "stroke"], errors="ignore")
y = df["stroke"].astype(int)

# Optional: scale for LR stability (RF doesn’t need it, but scaling won’t harm)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# Keep unscaled version for PDP plots (sklearn PDP can accept scaled; we’ll stick to scaled to match model inputs)
X_train_raw, X_test_raw = X.loc[X_train.index], X.loc[X_test.index]

# ---- 5) Train models ----
logreg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=300, random_state=42)

logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ---- 6) Evaluate & plot ----
def eval_and_plot(name, model, X_t, y_t, outstem):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_t)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X_t)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        y_prob = s
    else:
        raise ValueError("Model must support predict_proba or decision_function for AUC.")

    y_pred = model.predict(X_t)
    acc = accuracy_score(y_t, y_pred)
    auc = roc_auc_score(y_t, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Stroke", "Stroke"])
    disp.plot(values_format="d", cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrix_{outstem}.png"); plt.show()

    return acc, auc

acc_lr, auc_lr = eval_and_plot("Logistic Regression", logreg, X_test, y_test, "lr")
acc_rf, auc_rf = eval_and_plot("Random Forest", rf, X_test, y_test, "rf")

print(f"LogReg  - Acc: {acc_lr:.3f}, AUC: {auc_lr:.3f}")
print(f"RandFor - Acc: {acc_rf:.3f}, AUC: {auc_rf:.3f}")

# Performance bars
models = ["LogReg", "RandomForest"]
x = np.arange(len(models)); w = 0.35
plt.figure()
plt.bar(x - w/2, [acc_lr, acc_rf], w, label="Accuracy")
plt.bar(x + w/2, [auc_lr, auc_rf], w, label="AUC")
plt.xticks(x, models); plt.ylim(0, 1)
plt.ylabel("Score"); plt.title("Model Performance"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/performance_lr_vs_rf.png"); plt.show()

# ---- 7) Global importance: RF & LR ----
# RF feature importance
imp = rf.feature_importances_
idx_rf = np.argsort(imp)[::-1][:10]
plt.figure()
plt.barh(X.columns[idx_rf], imp[idx_rf])
plt.gca().invert_yaxis()
plt.xlabel("Importance"); plt.title("Top Features (RF)")
plt.tight_layout(); plt.savefig(f"{OUT}/feature_importance_rf.png"); plt.show()

# LR coefficients (absolute magnitude)
coef = logreg.coef_[0]
idx_lr = np.argsort(np.abs(coef))[::-1][:10]
plt.figure()
plt.barh(X.columns[idx_lr], np.abs(coef)[idx_lr])
plt.gca().invert_yaxis()
plt.xlabel("|Coefficient|"); plt.title("Top Features by |Coefficient| (LR)")
plt.tight_layout(); plt.savefig(f"{OUT}/coeff_importance_lr.png"); plt.show()

# ---- 8) PDPs (both models, using scaled X to match training) ----
# Choose features of clinical interest
feat_pair = ["age", "avg_glucose_level"]

fig, ax = plt.subplots(1, 2, figsize=(8, 4.5), dpi=150)
PartialDependenceDisplay.from_estimator(rf, X_train, features=feat_pair, ax=ax)
fig.suptitle("PDPs – RF (age, avg_glucose_level)")
plt.tight_layout(); fig.savefig(f"{OUT}/pdp_rf_age_glucose.png"); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(8, 4.5), dpi=150)
PartialDependenceDisplay.from_estimator(logreg, X_train, features=feat_pair, ax=ax)
fig.suptitle("PDPs – LR (age, avg_glucose_level)")
plt.tight_layout(); fig.savefig(f"{OUT}/pdp_lr_age_glucose.png"); plt.show()

# ---- 9) SHAP (global + local) for BOTH models ----
# Random Forest (TreeExplainer)
print("Computing SHAP for RF...")
rf_explainer = shap.TreeExplainer(rf)
try:
    rf_sv = rf_explainer(X_test)
    if hasattr(rf_sv, "values") and getattr(rf_sv.values, "ndim", 0) == 3:
        shap.summary_plot(rf_sv.values[:, :, 1], X_test, feature_names=X_test.columns, show=False)
    else:
        shap.summary_plot(rf_sv, X_test, feature_names=X_test.columns, show=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary_rf.png", bbox_inches="tight"); plt.show()

    if hasattr(rf_sv, "values") and getattr(rf_sv.values, "ndim", 0) == 3:
        ex_local_rf = shap.Explanation(
            values=rf_sv.values[0, :, 1],
            base_values=rf_sv.base_values[0, 1],
            data=X_test.iloc[0],
            feature_names=X_test.columns
        )
    else:
        ex_local_rf = shap.Explanation(
            values=rf_sv.values[0],
            base_values=np.mean(rf_sv.base_values),
            data=X_test.iloc[0],
            feature_names=X_test.columns
        )
    shap.plots.waterfall(ex_local_rf, show=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_waterfall_rf_first.png", bbox_inches="tight"); plt.show()
except Exception as e:
    print("RF SHAP fallback:", e)
    rf_sv_list = shap.TreeExplainer(rf).shap_values(X_test)
    rf_sv_use = rf_sv_list[1] if isinstance(rf_sv_list, list) else rf_sv_list
    shap.summary_plot(rf_sv_use, X_test, feature_names=X_test.columns, show=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary_rf.png", bbox_inches="tight"); plt.show()

# Logistic Regression (LinearExplainer -> fallback KernelExplainer)
print("Computing SHAP for LR...")
try:
    lr_explainer = shap.LinearExplainer(logreg, X_train, feature_perturbation="interventional")
    lr_sv = lr_explainer.shap_values(X_test)  # shape (n_samples, n_features)
    shap.summary_plot(lr_sv, X_test, feature_names=X_test.columns, show=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary_lr.png", bbox_inches="tight"); plt.show()

    ex_local_lr = shap.Explanation(
        values=lr_sv[0],
        base_values=lr_explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=X_test.columns
    )
    shap.plots.waterfall(ex_local_lr, show=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_waterfall_lr_first.png", bbox_inches="tight"); plt.show()
except Exception as e:
    print("LR SHAP fallback to KernelExplainer:", e)
    background = shap.sample(X_train, 200, random_state=42)
    lr_kexpl = shap.KernelExplainer(logreg.predict_proba, background)
    lr_sv = lr_kexpl.shap_values(X_test, nsamples=200)
    lr_sv_use = lr_sv[1] if isinstance(lr_sv, list) else lr_sv
    shap.summary_plot(lr_sv_use, X_test, feature_names=X_test.columns, show=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary_lr.png", bbox_inches="tight"); plt.show()

# ---- 10) LIME (local) for BOTH models ----
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=["No Stroke", "Stroke"],
    mode="classification",
    random_state=42
)

i = 0  # which test row to explain
x0 = X_test.iloc[i].values

def predict_proba_df(model, arr):
    df_arr = pd.DataFrame(arr, columns=X.columns)
    return model.predict_proba(df_arr)

# RF
exp_rf = lime_explainer.explain_instance(
    data_row=x0,
    predict_fn=lambda arr: predict_proba_df(rf, arr),
    num_features=5,
    top_labels=2
)
stroke_idx_rf = int(np.where(rf.classes_ == 1)[0][0])
available_rf = exp_rf.available_labels() if hasattr(exp_rf, "available_labels") else exp_rf.top_labels
label_plot_rf = stroke_idx_rf if stroke_idx_rf in available_rf else exp_rf.top_labels[0]
fig_rf = exp_rf.as_pyplot_figure(label=label_plot_rf)
plt.title("LIME – RF local explanation (Stroke)")
plt.tight_layout(); fig_rf.savefig(f"{OUT}/lime_local_rf_first.png"); plt.show()
exp_rf.save_to_file(f"{OUT}/lime_local_rf_first.html")

# LR
exp_lr = lime_explainer.explain_instance(
    data_row=x0,
    predict_fn=lambda arr: predict_proba_df(logreg, arr),
    num_features=5,
    top_labels=2
)
stroke_idx_lr = int(np.where(logreg.classes_ == 1)[0][0])
available_lr = exp_lr.available_labels() if hasattr(exp_lr, "available_labels") else exp_lr.top_labels
label_plot_lr = stroke_idx_lr if stroke_idx_lr in available_lr else exp_lr.top_labels[0]
fig_lr = exp_lr.as_pyplot_figure(label=label_plot_lr)
plt.title("LIME – LR local explanation (Stroke)")
plt.tight_layout(); fig_lr.savefig(f"{OUT}/lime_local_lr_first.png"); plt.show()
exp_lr.save_to_file(f"{OUT}/lime_local_lr_first.html")

# ---- 11) Quick summary ----
print("\n=== Summary for Slides ===")
print(f"Accuracy — LR: {acc_lr:.3f}, RF: {acc_rf:.3f}")
print(f"AUC      — LR: {auc_lr:.3f}, RF: {auc_rf:.3f}")
print("Top RF features:", ", ".join(X.columns[idx_rf][:5].tolist()))
print("Saved figures ->", os.path.abspath(OUT))
