
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------
# DATA DETECTIVE: STUDENT EARLY WARNING API
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "student_risk_model.joblib"

saved_artifact = joblib.load(MODEL_PATH)

model = saved_artifact["model"]
feature_columns = saved_artifact["feature_columns"]
train_medians = saved_artifact["train_medians"]

app = FastAPI(
    title="Data Detective Student Risk API",
    description="Explainable AI Student Early Warning System",
    version="1.0.0",
)


class StudentFeatures(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Student's numerical model features"
    )


@app.get("/")
def home():
    return {
        "project": "Data Detective",
        "status": "API is running",
        "model_loaded": True,
        "model_type": type(model).__name__,
        "number_of_features": len(feature_columns),
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "number_of_features": len(feature_columns),
        "required_features": list(feature_columns),
    }


@app.post("/predict")
def predict_student(data: StudentFeatures):
    try:
        supplied_features = data.features

        # Fill omitted features with training-set medians.
        # Missing features should ideally be supplied by
        # the upstream data-processing workflow.
        row = {}

        for feature in feature_columns:
            value = supplied_features.get(
                feature,
                train_medians.get(feature, 0)
            )
            row[feature] = value

        input_df = pd.DataFrame(
            [row],
            columns=feature_columns
        )

        # Reject invalid numerical values.
        if input_df.isnull().any().any():
            raise HTTPException(
                status_code=422,
                detail="Features contain missing values that cannot be filled."
            )

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        class_index = list(model.classes_).index(1)
        risk_probability = float(probabilities[class_index])

        outcome = (
            "Fail/Withdrawn"
            if prediction == 1
            else "Pass/Distinction"
        )

        return {
            "predicted_outcome": outcome,
            "predicted_class": prediction,
            "fail_withdrawn_probability": round(
                risk_probability, 4
            ),
            "fail_withdrawn_probability_percent": round(
                risk_probability * 100, 2
            ),
            "model_type": type(model).__name__,
            "features_used": len(feature_columns),
            "note": (
                "This is a model estimate, not a definitive "
                "outcome. Human review is required before "
                "any student support decision."
            ),
        }

    except HTTPException:
        raise
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature values: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        ) from exc
