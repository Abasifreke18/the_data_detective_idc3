
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

# =========================================================
# DATA DETECTIVE: STUDENT EARLY WARNING API
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "student_risk_model.joblib"
LIME_REFERENCE_PATH = BASE_DIR / "outputs" / "lime_reference_data.csv"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

saved_artifact = joblib.load(MODEL_PATH)

model = saved_artifact["model"]
feature_columns = list(saved_artifact["feature_columns"])
train_medians = saved_artifact["train_medians"]

app = FastAPI(
    title="Data Detective Student Risk API",
    description=(
        "Evidence-first student early warning API "
        "with Random Forest predictions and XAI endpoints."
    ),
    version="2.0.0",
)


class StudentFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        ...,
        description="All 15 numerical features required by the model."
    )


def prepare_features(data: StudentFeatures) -> pd.DataFrame:
    """Validate and order the complete model feature vector."""
    supplied = data.features

    missing = [name for name in feature_columns if name not in supplied]
    extra = [name for name in supplied if name not in feature_columns]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing required model features",
                "missing_features": missing,
            },
        )

    if extra:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Unexpected model features",
                "unexpected_features": extra,
            },
        )

    try:
        values = [float(supplied[name]) for name in feature_columns]
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="All model features must be numerical.",
        ) from exc

    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422,
            detail="Features cannot contain NaN or infinite values.",
        )

    return pd.DataFrame([values], columns=feature_columns)


def get_fail_probability(input_df: pd.DataFrame) -> float:
    """Return the probability of the model's class 1."""
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        raise HTTPException(
            status_code=500,
            detail="Model does not contain the expected class 1.",
        )

    return float(probabilities[classes.index(1)])


def get_shap_values(input_df: pd.DataFrame) -> np.ndarray:
    """Return SHAP contributions for the Fail/Withdrawn class."""
    try:
        import shap
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="SHAP is unavailable. Install the shap dependency.",
        ) from exc

    try:
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(input_df)

        # SHAP versions may return a list by class or a NumPy array.
        if isinstance(raw_values, list):
            classes = list(model.classes_)
            if 1 not in classes:
                raise ValueError("Class 1 is unavailable.")
            values = np.asarray(raw_values[classes.index(1)])
        else:
            values = np.asarray(raw_values)

            if values.ndim == 3:
                classes = list(model.classes_)
                values = values[:, :, classes.index(1)]

        if values.ndim == 2:
            values = values[0]

        if values.shape[0] != len(feature_columns):
            raise ValueError(
                "Unexpected SHAP output shape; feature mapping failed."
            )

        return values.astype(float)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {exc}",
        ) from exc


@app.get("/")
def home():
    return {
        "project": "Data Detective",
        "status": "API is running",
        "model_loaded": True,
        "model_type": type(model).__name__,
        "number_of_features": len(feature_columns),
        "explanation_endpoints": [
            "/explain/shap",
            "/explain/lime",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "number_of_features": len(feature_columns),
        "required_features": feature_columns,
        "target_definition": saved_artifact.get(
            "target_definition",
            "Class 1: Fail/Withdrawn; Class 0: Pass/Distinction",
        ),
        "prediction_cutoff_days": saved_artifact.get(
            "prediction_cutoff_days",
            60,
        ),
        "lime_reference_available": LIME_REFERENCE_PATH.exists(),
    }


@app.post("/predict")
def predict_student(data: StudentFeatures):
    try:
        input_df = prepare_features(data)

        prediction = int(model.predict(input_df)[0])
        risk_probability = get_fail_probability(input_df)

        outcome = (
            "Fail/Withdrawn"
            if prediction == 1
            else "Pass/Distinction"
        )

        return {
            "predicted_outcome": outcome,
            "predicted_class": prediction,
            "fail_withdrawn_probability": round(risk_probability, 4),
            "fail_withdrawn_probability_percent": round(
                risk_probability * 100, 2
            ),
            "model_type": type(model).__name__,
            "features_used": len(feature_columns),
            "note": (
                "This is a model estimate, not a definitive outcome. "
                "Human review is required before student-support decisions."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.post("/explain/shap")
def explain_shap(data: StudentFeatures):
    """Explain a prediction using SHAP contributions."""
    input_df = prepare_features(data)
    probability = get_fail_probability(input_df)
    contributions = get_shap_values(input_df)

    ranked = sorted(
        [
            {
                "feature": feature,
                "value": float(input_df.iloc[0][feature]),
                "shap_value": float(contribution),
            }
            for feature, contribution in zip(
                feature_columns, contributions
            )
        ],
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    return {
        "method": "SHAP TreeExplainer",
        "explained_class": "Fail/Withdrawn",
        "fail_withdrawn_probability": round(probability, 4),
        "top_contributions": ranked[:10],
        "interpretation_note": (
            "SHAP contributions explain this model output. "
            "They do not establish causation."
        ),
    }


@app.post("/explain/lime")
def explain_lime(data: StudentFeatures):
    """Explain an individual prediction using LIME."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="LIME is unavailable. Install the lime dependency.",
        ) from exc

    if not LIME_REFERENCE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "LIME reference data is not available. Export a "
                "representative reference sample from the actual training "
                "features to outputs/lime_reference_data.csv. "
                "Do not use fabricated student records."
            ),
        )

    try:
        reference = pd.read_csv(LIME_REFERENCE_PATH)

        missing_columns = [
            name for name in feature_columns
            if name not in reference.columns
        ]
        if missing_columns:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "LIME reference data has missing columns",
                    "missing_features": missing_columns,
                },
            )

        reference = reference[feature_columns].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(reference) < 20:
            raise HTTPException(
                status_code=503,
                detail=(
                    "LIME needs a larger valid reference sample "
                    "from the actual training features."
                ),
            )

        input_df = prepare_features(data)
        input_values = input_df.iloc[0].to_numpy(dtype=float)

        explainer = LimeTabularExplainer(
            training_data=reference.to_numpy(dtype=float),
            feature_names=feature_columns,
            class_names=["Pass/Distinction", "Fail/Withdrawn"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

        explanation = explainer.explain_instance(
            data_row=input_values,
            predict_fn=lambda values: model.predict_proba(
                pd.DataFrame(values, columns=feature_columns)
            ),
            labels=(list(model.classes_).index(1),),
            num_features=min(10, len(feature_columns)),
            num_samples=3000,
        )

        contributions = [
            {
                "condition": condition,
                "contribution": float(weight),
            }
            for condition, weight in explanation.as_list(
                label=list(model.classes_).index(1)
            )
        ]

        return {
            "method": "LIME Tabular Explainer",
            "explained_class": "Fail/Withdrawn",
            "fail_withdrawn_probability": round(
                get_fail_probability(input_df), 4
            ),
            "local_contributions": contributions,
            "interpretation_note": (
                "LIME approximates model behaviour locally. "
                "Contributions are not causal evidence."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"LIME explanation failed: {exc}",
        ) from exc
