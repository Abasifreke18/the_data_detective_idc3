
# ============================================================
# DATA DETECTIVE — IDC 3.0
# STAGE 3: STUDENT RISK MODEL WITH VLE ENGAGEMENT
# OULAD | BBB | 2013J | DAY-60 PREDICTION CUTOFF
# ============================================================

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

COURSE_MODULE = "BBB"
COURSE_PRESENTATION = "2013J"
CUTOFF_DAY = 60
RANDOM_STATE = 42
CHUNK_SIZE = 500_000

STUDENT_FILE = DATA_DIR / "studentInfo.csv"
REGISTRATION_FILE = DATA_DIR / "studentRegistration.csv"
ASSESSMENTS_FILE = DATA_DIR / "assessments.csv"
STUDENT_ASSESSMENT_FILE = DATA_DIR / "studentAssessment.csv"
VLE_FILE = DATA_DIR / "studentVle.csv"

required_files = [
    STUDENT_FILE,
    REGISTRATION_FILE,
    ASSESSMENTS_FILE,
    STUDENT_ASSESSMENT_FILE,
    VLE_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

print("\n" + "=" * 65)
print("DATA DETECTIVE — DAY-60 EARLY WARNING MODEL")
print("Adding VLE engagement features")
print("=" * 65)

# Preserve the previous model outputs before replacing them.
BACKUP_DIR = OUTPUT_DIR / "baseline_before_vle"
BACKUP_DIR.mkdir(exist_ok=True)

files_to_backup = [
    "data_quality_report.csv",
    "student_feature_table.csv",
    "X_train.csv",
    "X_test.csv",
    "y_test.csv",
    "baseline_metrics.json",
    "student_risk_model.joblib",
    "test_predictions.csv",
]

for filename in files_to_backup:
    source = OUTPUT_DIR / filename
    destination = BACKUP_DIR / filename

    if source.exists() and not destination.exists():
        shutil.copy2(source, destination)

print(f"\nPrevious outputs preserved in: {BACKUP_DIR}")

# ------------------------------------------------------------
# 2. LOAD CORE DATA
# ------------------------------------------------------------

print("\n[1/7] Loading student and assessment data...")

students = pd.read_csv(STUDENT_FILE)
registrations = pd.read_csv(REGISTRATION_FILE)
assessments = pd.read_csv(ASSESSMENTS_FILE)
student_assessments = pd.read_csv(STUDENT_ASSESSMENT_FILE)

print(f"All students: {len(students):,}")
print(f"Assessment records: {len(student_assessments):,}")
print(f"Assessment definitions: {len(assessments):,}")
print(f"Registration records: {len(registrations):,}")

# Standardise numeric identifiers and date fields.
students["id_student"] = pd.to_numeric(
    students["id_student"], errors="coerce"
)
registrations["id_student"] = pd.to_numeric(
    registrations["id_student"], errors="coerce"
)
student_assessments["id_student"] = pd.to_numeric(
    student_assessments["id_student"], errors="coerce"
)

registrations["date_registration"] = pd.to_numeric(
    registrations["date_registration"], errors="coerce"
)
registrations["date_unregistration"] = pd.to_numeric(
    registrations["date_unregistration"], errors="coerce"
)

assessments["date"] = pd.to_numeric(
    assessments["date"], errors="coerce"
)
student_assessments["date_submitted"] = pd.to_numeric(
    student_assessments["date_submitted"], errors="coerce"
)
student_assessments["score"] = pd.to_numeric(
    student_assessments["score"], errors="coerce"
)

# ------------------------------------------------------------
# 3. DEFINE THE ELIGIBLE DAY-60 COHORT
# ------------------------------------------------------------

print("\n[2/7] Preparing the day-60 student cohort...")

course_students = students[
    (students["code_module"] == COURSE_MODULE)
    & (students["code_presentation"] == COURSE_PRESENTATION)
].copy()

course_registrations = registrations[
    (registrations["code_module"] == COURSE_MODULE)
    & (
        registrations["code_presentation"]
        == COURSE_PRESENTATION
    )
].copy()

# A student must have a known registration date no later than
# the cutoff and must not have withdrawn before day 60.
eligible_registrations = course_registrations[
    course_registrations["date_registration"].notna()
    & (course_registrations["date_registration"] <= CUTOFF_DAY)
    & (
        course_registrations["date_unregistration"].isna()
        | (
            course_registrations["date_unregistration"]
            >= CUTOFF_DAY
        )
    )
].copy()

# Use the registration table as the source of registration dates.
cohort = course_students.merge(
    eligible_registrations[
        [
            "id_student",
            "date_registration",
            "date_unregistration",
        ]
    ],
    on="id_student",
    how="inner",
    validate="one_to_one",
)

# Target: 1 = Fail/Withdrawn; 0 = Pass/Distinction.
target_map = {
    "Fail": 1,
    "Withdrawn": 1,
    "Pass": 0,
    "Distinction": 0,
}

cohort["target"] = cohort["final_result"].map(target_map)
cohort = cohort[cohort["target"].notna()].copy()
cohort["target"] = cohort["target"].astype(int)

if cohort.empty:
    raise ValueError(
        "No eligible students remain. Check the course filters "
        "and registration dates."
    )

print(f"Eligible students: {len(cohort):,}")
print("\nTarget distribution:")
print(
    cohort["target"]
    .map({0: "Pass/Distinction", 1: "Fail/Withdrawn"})
    .value_counts()
    .to_string()
)

# ------------------------------------------------------------
# 4. ASSESSMENT FEATURES AVAILABLE BY DAY 60
# ------------------------------------------------------------

print("\n[3/7] Building assessment features...")

course_assessments = assessments[
    (assessments["code_module"] == COURSE_MODULE)
    & (
        assessments["code_presentation"]
        == COURSE_PRESENTATION
    )
    & assessments["date"].notna()
    & (assessments["date"] <= CUTOFF_DAY)
].copy()

due_assessment_ids = course_assessments[
    ["id_assessment", "date"]
].rename(columns={"date": "due_date"})

due_count = len(due_assessment_ids)

print(f"Assessments due by day {CUTOFF_DAY}: {due_count}")

submissions = student_assessments.merge(
    due_assessment_ids,
    on="id_assessment",
    how="inner",
)

# Only submissions actually recorded by the prediction cutoff.
submissions = submissions[
    submissions["date_submitted"].notna()
    & (submissions["date_submitted"] <= CUTOFF_DAY)
].copy()

# A valid assessment score must be within 0–100.
submissions["valid_score"] = submissions["score"].where(
    submissions["score"].between(0, 100)
)

submissions["days_after_due"] = (
    submissions["date_submitted"] - submissions["due_date"]
)

submissions["is_late"] = (
    submissions["date_submitted"] > submissions["due_date"]
).astype(int)

assessment_features = (
    submissions.groupby("id_student")
    .agg(
        assessments_submitted=("id_assessment", "nunique"),
        mean_assessment_score=("valid_score", "mean"),
        min_assessment_score=("valid_score", "min"),
        valid_assessment_scores=("valid_score", "count"),
        mean_days_after_due=("days_after_due", "mean"),
        late_submissions=("is_late", "sum"),
    )
    .reset_index()
)

assessment_features["assessments_due_by_cutoff"] = due_count

assessment_features["missing_assessment_scores"] = (
    due_count - assessment_features["valid_assessment_scores"]
).clip(lower=0)

assessment_features["assessments_submitted_by_cutoff"] = (
    assessment_features["assessments_submitted"]
)

assessment_features["has_assessment_evidence"] = (
    assessment_features["assessments_submitted"] > 0
).astype(int)

assessment_features = assessment_features.drop(
    columns=["valid_assessment_scores"]
)

# ------------------------------------------------------------
# 5. STREAM THE LARGE VLE FILE AND BUILD ENGAGEMENT FEATURES
# ------------------------------------------------------------

print("\n[4/7] Processing VLE activity in chunks...")
print("This step may take several minutes.")

# Per-student accumulators avoid loading the entire VLE dataset.
click_totals = {}
activity_record_totals = {}
active_dates = {}
resource_ids = {}

vle_rows_processed = 0
vle_rows_in_course = 0
vle_rows_by_cutoff = 0

vle_reader = pd.read_csv(
    VLE_FILE,
    chunksize=CHUNK_SIZE,
    usecols=[
        "code_module",
        "code_presentation",
        "id_student",
        "id_site",
        "date",
        "sum_click",
    ],
)

for chunk_number, chunk in enumerate(vle_reader, start=1):
    vle_rows_processed += len(chunk)

    chunk = chunk[
        (chunk["code_module"] == COURSE_MODULE)
        & (
            chunk["code_presentation"]
            == COURSE_PRESENTATION
        )
    ].copy()

    vle_rows_in_course += len(chunk)

    chunk["id_student"] = pd.to_numeric(
        chunk["id_student"], errors="coerce"
    )
    chunk["date"] = pd.to_numeric(
        chunk["date"], errors="coerce"
    )
    chunk["sum_click"] = pd.to_numeric(
        chunk["sum_click"], errors="coerce"
    )

    chunk = chunk[
        chunk["id_student"].notna()
        & chunk["date"].notna()
        & (chunk["date"] <= CUTOFF_DAY)
    ].copy()

    vle_rows_by_cutoff += len(chunk)

    if chunk.empty:
        print(
            f"  Chunk {chunk_number}: no matching rows "
            f"at or before cutoff"
        )
        continue

    chunk["id_student"] = chunk["id_student"].astype(int)
    chunk["sum_click"] = chunk["sum_click"].fillna(0)

    # Aggregate clicks and raw activity-record counts.
    totals = chunk.groupby("id_student").agg(
        clicks=("sum_click", "sum"),
        records=("sum_click", "size"),
    )

    for student_id, row in totals.iterrows():
        student_id = int(student_id)

        click_totals[student_id] = (
            click_totals.get(student_id, 0)
            + float(row["clicks"])
        )

        activity_record_totals[student_id] = (
            activity_record_totals.get(student_id, 0)
            + int(row["records"])
        )

    # Use sets to avoid counting the same date or resource
    # multiple times across different chunks.
    unique_student_dates = chunk[
        ["id_student", "date"]
    ].drop_duplicates()

    for student_id, date in unique_student_dates.itertuples(
        index=False, name=None
    ):
        student_id = int(student_id)
        active_dates.setdefault(student_id, set()).add(
            int(date)
        )

    unique_student_resources = chunk[
        ["id_student", "id_site"]
    ].dropna().drop_duplicates()

    for student_id, site_id in unique_student_resources.itertuples(
        index=False, name=None
    ):
        student_id = int(student_id)
        resource_ids.setdefault(student_id, set()).add(
            int(site_id)
        )

    print(
        f"  Processed chunk {chunk_number}; "
        f"rows scanned: {vle_rows_processed:,}"
    )

print(f"VLE rows scanned: {vle_rows_processed:,}")
print(f"Rows for selected course: {vle_rows_in_course:,}")
print(f"Course rows at or before day {CUTOFF_DAY}: "
      f"{vle_rows_by_cutoff:,}")

vle_features = pd.DataFrame({
    "id_student": list(click_totals.keys()),
    "total_vle_clicks": [
        click_totals[k] for k in click_totals
    ],
    "vle_activity_records": [
        activity_record_totals[k] for k in click_totals
    ],
    "active_learning_days": [
        len(active_dates.get(k, set())) for k in click_totals
    ],
    "unique_learning_resources": [
        len(resource_ids.get(k, set())) for k in click_totals
    ],
})

print(f"Students with VLE activity: {len(vle_features):,}")

# ------------------------------------------------------------
# 6. ASSEMBLE THE FEATURE TABLE
# ------------------------------------------------------------

print("\n[5/7] Combining student, assessment and VLE features...")

base_columns = [
    "id_student",
    "num_of_prev_attempts",
    "studied_credits",
    "target",
    "date_registration",
]

feature_table = cohort[base_columns].copy()

feature_table = feature_table.merge(
    assessment_features,
    on="id_student",
    how="left",
    validate="one_to_one",
)

feature_table = feature_table.merge(
    vle_features,
    on="id_student",
    how="left",
    validate="one_to_one",
)

# No recorded activity is represented as zero activity.
vle_feature_columns = [
    "total_vle_clicks",
    "vle_activity_records",
    "active_learning_days",
    "unique_learning_resources",
]

feature_table[vle_feature_columns] = (
    feature_table[vle_feature_columns].fillna(0)
)

feature_table["assessments_submitted"] = (
    feature_table["assessments_submitted"].fillna(0)
)
feature_table["assessments_submitted_by_cutoff"] = (
    feature_table["assessments_submitted_by_cutoff"].fillna(0)
)
feature_table["assessments_due_by_cutoff"] = (
    feature_table["assessments_due_by_cutoff"].fillna(due_count)
)
feature_table["missing_assessment_scores"] = (
    feature_table["missing_assessment_scores"].fillna(due_count)
)
feature_table["late_submissions"] = (
    feature_table["late_submissions"].fillna(0)
)
feature_table["has_assessment_evidence"] = (
    feature_table["has_assessment_evidence"].fillna(0)
)

# Retain student identifiers and labels for audit purposes,
# but never use them as predictive features.
feature_columns = [
    "num_of_prev_attempts",
    "studied_credits",
    "assessments_submitted",
    "mean_assessment_score",
    "min_assessment_score",
    "missing_assessment_scores",
    "mean_days_after_due",
    "late_submissions",
    "assessments_due_by_cutoff",
    "assessments_submitted_by_cutoff",
    "has_assessment_evidence",
    "total_vle_clicks",
    "vle_activity_records",
    "active_learning_days",
    "unique_learning_resources",
]

feature_table = feature_table.replace(
    [np.inf, -np.inf], np.nan
)

feature_table.to_csv(
    OUTPUT_DIR / "student_feature_table.csv",
    index=False,
)

print(f"Feature table rows: {len(feature_table):,}")
print(f"Predictive features: {len(feature_columns)}")

print("\nVLE engagement feature preview:")
print(
    feature_table[
        [
            "total_vle_clicks",
            "active_learning_days",
            "unique_learning_resources",
            "vle_activity_records",
        ]
    ].describe().round(2).to_string()
)

# ------------------------------------------------------------
# 7. TRAIN, EVALUATE AND SAVE THE MODEL
# ------------------------------------------------------------

print("\n[6/7] Training and evaluating Random Forest...")

X = feature_table[feature_columns].copy()
y = feature_table["target"].astype(int).copy()

# Split before imputing so medians come from training data only.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

train_medians = X_train.median().fillna(0)

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

model = RandomForestClassifier(
    n_estimators=250,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

positive_class_index = list(model.classes_).index(1)
risk_probabilities = probabilities[:, positive_class_index]

metrics = {
    "course_module": COURSE_MODULE,
    "course_presentation": COURSE_PRESENTATION,
    "prediction_cutoff_days": CUTOFF_DAY,
    "target_definition": (
        "Fail/Withdrawn=1; Pass/Distinction=0"
    ),
    "eligible_students": int(len(feature_table)),
    "training_records": int(len(X_train)),
    "test_records": int(len(X_test)),
    "number_of_features": int(len(feature_columns)),
    "feature_columns": feature_columns,
    "vle_rows_scanned": int(vle_rows_processed),
    "vle_course_rows": int(vle_rows_in_course),
    "vle_rows_at_or_before_cutoff": int(vle_rows_by_cutoff),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_test, predictions)
    ),
    "precision_positive_class": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall_positive_class": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1_positive_class": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, risk_probabilities)
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist(),
    "validation_note": (
        "Preliminary single stratified holdout evaluation; "
        "not yet externally or temporally validated."
    ),
}

print("\nModel metrics:")
for key in [
    "accuracy",
    "balanced_accuracy",
    "precision_positive_class",
    "recall_positive_class",
    "f1_positive_class",
    "roc_auc",
]:
    print(f"{key}: {metrics[key]:.4f}")

print("\nConfusion matrix (actual rows, predicted columns):")
print(np.array(metrics["confusion_matrix"]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Pass/Distinction", "Fail/Withdrawn"],
        zero_division=0,
    )
)

# Save training and testing matrices with consistent feature order.
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
y_test.to_frame(name="target").to_csv(
    OUTPUT_DIR / "y_test.csv", index=False
)

# Preserve the actual test targets and predicted probabilities.
test_predictions = pd.DataFrame({
    "id_student": feature_table.loc[
        X_test.index, "id_student"
    ].to_numpy(),
    "actual_target": y_test.to_numpy(),
    "predicted_target": predictions,
    "predicted_fail_withdrawn_probability": risk_probabilities,
})

test_predictions.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "baseline_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

# Save the model in the same dictionary format expected by
# explain_model.py.
model_artifact = {
    "model": model,
    "feature_columns": feature_columns,
    "train_medians": train_medians.to_dict(),
    "target_definition": metrics["target_definition"],
    "course_module": COURSE_MODULE,
    "course_presentation": COURSE_PRESENTATION,
    "prediction_cutoff_days": CUTOFF_DAY,
}

dump(
    model_artifact,
    OUTPUT_DIR / "student_risk_model.joblib",
)

# Data quality report includes raw missingness and duplicates.
quality_files = [
    ("studentInfo.csv", STUDENT_FILE),
    ("studentRegistration.csv", REGISTRATION_FILE),
    ("assessments.csv", ASSESSMENTS_FILE),
    ("studentAssessment.csv", STUDENT_ASSESSMENT_FILE),
    ("studentVle.csv", VLE_FILE),
]

quality_rows = []

for filename, path in quality_files:
    if filename == "studentVle.csv":
        # The VLE file is large; use the streaming scan count.
        quality_rows.append({
            "file": filename,
            "rows": vle_rows_processed,
            "duplicate_rows": "Not computed",
            "missing_cells": "Not computed",
            "columns": 6,
            "note": (
                "Streamed in chunks; full-file duplicate and "
                "missing-cell audit not computed."
            ),
        })
        continue

    frame = pd.read_csv(path)

    quality_rows.append({
        "file": filename,
        "rows": len(frame),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "columns": len(frame.columns),
        "note": "Raw CSV checks; semantic validation is separate.",
    })

pd.DataFrame(quality_rows).to_csv(
    OUTPUT_DIR / "data_quality_report.csv",
    index=False,
)

# Feature importance from the trained Random Forest.
importance = pd.DataFrame({
    "feature": feature_columns,
    "random_forest_importance": model.feature_importances_,
}).sort_values(
    "random_forest_importance", ascending=False
)

importance.to_csv(
    OUTPUT_DIR / "random_forest_feature_importance.csv",
    index=False,
)

print("\nRandom Forest feature importance:")
print(importance.to_string(index=False))

print("\n[7/7] Training completed.")
print(f"Outputs saved in: {OUTPUT_DIR}")
print(f"Backup saved in: {BACKUP_DIR}")

print("\nImportant output files:")
for filename in [
    "student_feature_table.csv",
    "data_quality_report.csv",
    "baseline_metrics.json",
    "student_risk_model.joblib",
    "test_predictions.csv",
    "random_forest_feature_importance.csv",
    "X_train.csv",
    "X_test.csv",
    "y_test.csv",
]:
    print(f" - {filename}")

print(
    "\nReminder: results are preliminary. "
    "Predictions should support human review, not determine "
    "student outcomes automatically."
)
