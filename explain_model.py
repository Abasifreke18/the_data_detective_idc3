
# ============================================================
# DATA DETECTIVE — IDC 3.0
# STAGE 2: EXPLAINABLE AI USING SHAP AND LIME
# Student Early Warning System
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from lime.lime_tabular import LimeTabularExplainer
from joblib import load

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. SET UP PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "student_risk_model.joblib"
X_TRAIN_PATH = OUTPUT_DIR / "X_train.csv"
X_TEST_PATH = OUTPUT_DIR / "X_test.csv"

print("\n" + "=" * 60)
print("DATA DETECTIVE: SHAP AND LIME EXPLAINABILITY")
print("=" * 60)

# ------------------------------------------------------------
# 2. LOAD THE TRAINED MODEL AND DATA
# ------------------------------------------------------------

print("\n[1/5] Loading trained model and datasets...")

saved_artifact = load(MODEL_PATH)

model = saved_artifact["model"]
feature_names_from_model = saved_artifact["feature_columns"]

print("Loaded model:", type(model).__name__)
print("Model features:", len(feature_names_from_model))
X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

# Keep feature names and order consistent with training.
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
    X_train = X_train[feature_names]
    X_test = X_test[feature_names]
else:
    feature_names = list(X_train.columns)
    X_test = X_test[feature_names]

# Convert to numeric and handle any missing or infinite values.
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Use training medians to fill missing values consistently.
medians = X_train.median().fillna(0)
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")
print(f"Features: {len(feature_names)}")

# ------------------------------------------------------------
# 3. SHAP: GLOBAL FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\n[2/5] Calculating SHAP explanations...")

# Explain a manageable sample of the test set.
sample_size = min(200, len(X_test))
X_explain = X_test.iloc[:sample_size].copy()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_explain)

# Support SHAP output formats from different library versions.
if isinstance(shap_values, list):
    # For binary classification, index 1 is the positive class.
    values = (
        shap_values[1]
        if len(shap_values) > 1
        else shap_values[0]
    )
else:
    values = np.asarray(shap_values)

    if values.ndim == 3:
        # Typical shape: rows x features x classes.
        positive_class_index = 1 if values.shape[2] > 1 else 0
        values = values[:, :, positive_class_index]

values = np.asarray(values)

if values.shape != (len(X_explain), len(feature_names)):
    raise ValueError(
        "Unexpected SHAP output shape: "
        f"{values.shape}. Expected "
        f"{(len(X_explain), len(feature_names))}."
    )

# Calculate average absolute SHAP value for each feature.
importance = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": np.abs(values).mean(axis=0)
}).sort_values("mean_absolute_shap", ascending=False)

importance.to_csv(
    OUTPUT_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nTop SHAP features:")
print(importance.head(10).to_string(index=False))

# ------------------------------------------------------------
# 4. SHAP: SUMMARY AND BAR PLOTS
# ------------------------------------------------------------

print("\n[3/5] Creating SHAP visualisations...")

plt.figure()
shap.summary_plot(
    values,
    X_explain,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close("all")

plt.figure()
shap.summary_plot(
    values,
    X_explain,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_importance_bar.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close("all")

# Save SHAP values for individual test records.
shap_records = pd.DataFrame(values, columns=feature_names)
shap_records.insert(
    0, "test_record_index", X_explain.index.to_numpy()
)
shap_records.to_csv(
    OUTPUT_DIR / "shap_individual_explanations.csv",
    index=False
)

# ------------------------------------------------------------
# 5. LIME: EXPLAIN ONE HIGH-RISK PREDICTION
# ------------------------------------------------------------

print("\n[4/5] Generating a LIME explanation...")

# Find the test record with the highest predicted probability
# of belonging to the Fail/Withdrawn class.
probabilities = model.predict_proba(X_test)

classes = list(model.classes_)
if 1 not in classes:
    raise ValueError(
        "The trained model does not contain positive class 1."
    )

positive_class_index = classes.index(1)
risk_probabilities = probabilities[:, positive_class_index]
highest_risk_position = int(np.argmax(risk_probabilities))

student_row = X_test.iloc[highest_risk_position].to_numpy()
student_index = X_test.index[highest_risk_position]
risk_probability = float(
    risk_probabilities[highest_risk_position]
)

print(f"Selected test record index: {student_index}")
print(f"Predicted Fail/Withdrawn probability: "
      f"{risk_probability:.3f}")

lime_explainer = LimeTabularExplainer(
    training_data=X_train.to_numpy(dtype=float),
    feature_names=feature_names,
    class_names=["Pass/Distinction", "Fail/Withdrawn"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

lime_explanation = lime_explainer.explain_instance(
    data_row=student_row,
    predict_fn=model.predict_proba,
    num_features=len(feature_names),
    num_samples=3000
)

lime_items = lime_explanation.as_list(
    label=positive_class_index
)

lime_df = pd.DataFrame(
    lime_items,
    columns=["feature_condition", "contribution_to_class_1"]
)

lime_df.to_csv(
    OUTPUT_DIR / "lime_highest_risk_explanation.csv",
    index=False
)

lime_explanation.save_to_file(
    str(OUTPUT_DIR / "lime_highest_risk_explanation.html")
)

print("\nLIME explanation for the selected record:")
print(lime_df.to_string(index=False))

# Save the selected record and its predicted probability.
selected_record = {
    "test_record_index": int(student_index),
    "predicted_fail_withdrawn_probability": risk_probability,
    "predicted_class": int(model.predict(
        X_test.iloc[[highest_risk_position]]
    )[0]),
    "explanation_method": "LIME",
    "warning": (
        "This is a model prediction, not a diagnosis or proof "
        "of the causes of a student's academic outcome."
    )
}

with open(
    OUTPUT_DIR / "explainability_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(selected_record, file, indent=4)

# ------------------------------------------------------------
# 6. COMPLETION SUMMARY
# ------------------------------------------------------------

print("\n[5/5] Explainability analysis completed.")

print("\nFiles saved in:", OUTPUT_DIR)
print(" - shap_feature_importance.csv")
print(" - shap_summary.png")
print(" - shap_importance_bar.png")
print(" - shap_individual_explanations.csv")
print(" - lime_highest_risk_explanation.csv")
print(" - lime_highest_risk_explanation.html")
print(" - explainability_summary.json")

print("\nImportant interpretation note:")
print(
    "SHAP and LIME explain model behaviour. "
    "They do not establish causation."
)
print("\nNext step: inspect the generated outputs.")
