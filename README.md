# Data Detective IDC3
### Evidence-First Student Early Warning System

**Beyond Automation — Data Powering AI Agents | Ibom Data Connect 3.0**

Data Detective is an AI-powered student early warning system designed to identify potential academic risks from learning analytics data, explain model predictions, and refer cases with insufficient evidence for human review.

Rather than generating predictions without context, Data Detective combines machine learning, evidence validation, explainable AI, and workflow automation to support more transparent and responsible student intervention.

**Project repository:** https://github.com/Abasifreke18/the_data_detective_idc3  
**Live API:** https://the-data-detective-idc3.onrender.com  
**API documentation:** https://the-data-detective-idc3.onrender.com/docs

---

## 1. The Problem

Academic difficulties can develop before they become visible in final examination results. Educators need ways to identify students who may require additional support using the learning evidence available to them.

However, a prediction alone is not enough. An effective early warning system should also consider whether the available evidence is sufficient, explain the factors influencing its predictions, and avoid treating model outputs as confirmed outcomes.

Data Detective addresses this challenge by combining student risk prediction with evidence validation, explainability, and a human-review pathway.

## 2. Our Solution

Data Detective processes student assessment and learning activity data to estimate the likelihood of a student receiving a **Fail or Withdrawn** outcome.

The system:

- Prepares learning analytics data using a day-60 observation cutoff.
- Builds the features required by a trained machine learning model.
- Validates the availability of the required evidence.
- Predicts potential academic outcomes for eligible records.
- Uses SHAP and LIME to explain model predictions.
- Routes cases with insufficient evidence for human review.
- Integrates the prediction and explanation APIs into an automated n8n workflow.

The goal is to support earlier, evidence-informed intervention while keeping educators involved in important decisions.

## 3. How It Works

```text
Student Learning Analytics Data
              |
              v
      Data Preparation
       (Day-60 Cutoff)
              |
              v
      Feature Engineering
              |
              v
      Evidence Validation
              |
        +-----+------+
        |            |
        v            v
  Sufficient       Insufficient
   Evidence          Evidence
        |            |
        v            v
  Risk Prediction  Human Review
        |
        v
   SHAP + LIME
   Explanations
        |
        v
  Results for Human
      Assessment
```

The workflow is orchestrated using n8n, with a deployed API serving the machine learning model and explainability endpoints.

## 4. Dataset

The project uses the **Open University Learning Analytics Dataset (OULAD)**, which contains student demographic, assessment, registration, and virtual learning environment activity data.

The current workflow uses the BBB/2013J cohort and applies a day-60 cutoff when constructing student features.

Relevant data includes:

- Previous assessment attempts and studied credits.
- Assessment scores and missing assessment evidence.
- Late submissions and submission timing.
- Virtual learning environment clicks and activity records.
- Active learning days and unique learning resources.
- Registration and unregistration information.

The system uses the available data to construct 15 model features. Cases that do not meet the evidence requirements are routed for review rather than automatically scored.

Dataset source: https://analyse.kmi.open.ac.uk/open_dataset

## 5. Machine Learning Model

Data Detective uses a **Random Forest classifier** to estimate academic outcome risk.

The model distinguishes between two outcome classes:

| Class | Meaning |
|---|---|
| 0 | Pass / Distinction |
| 1 | Fail / Withdrawn |

The model returns a predicted class and estimated class probabilities. These probabilities represent model estimates, not guarantees about a student's future performance.

### Evaluation results

The model was evaluated on a held-out test set of 357 records.

| Metric | Result |
|---|---:|
| Accuracy | 75.63% |
| Balanced accuracy | 73.67% |
| Precision — Fail / Withdrawn | 71.65% |
| Recall — Fail / Withdrawn | 64.08% |
| F1 score — Fail / Withdrawn | 67.66% |
| ROC-AUC | 80.59% |

The saved model was also checked against the held-out predictions, with a maximum prediction difference of approximately \(2.2 \times 10^{-16}\).

These results describe performance on the evaluated test set and do not guarantee equivalent performance in other cohorts or institutions.

## 6. Explainable AI: SHAP and LIME

Data Detective incorporates two explainability methods to help users inspect model outputs.

**SHAP (SHapley Additive exPlanations)**  
Provides feature contribution estimates that help explain how individual features influence a model prediction.

**LIME (Local Interpretable Model-agnostic Explanations)**  
Provides a local explanation of an individual prediction by approximating model behaviour around that specific example.

Together, these methods help make predictions easier to investigate and question.

Explanations are not proof of causation, and feature contributions should not be interpreted as evidence that a particular factor caused a student to struggle academically.

## 7. API Endpoints

The deployed API exposes the following endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Checks API health and model loading status. |
| `GET /model-info` | Returns information about the model. |
| `POST /predict` | Generates a student outcome prediction and probabilities. |
| `POST /explain/shap` | Generates SHAP-based explanation results. |
| `POST /explain/lime` | Generates LIME-based local explanation results. |

### Access the API

- **Base URL:** https://the-data-detective-idc3.onrender.com
- **Interactive API documentation:** https://the-data-detective-idc3.onrender.com/docs
- **Health check:** https://the-data-detective-idc3.onrender.com/health

The prediction endpoint expects the 15 features defined by the project's feature-engineering pipeline. Use the interactive API documentation to inspect the current request schema and response format.

## 8. Workflow Automation with n8n

The n8n workflow connects data preparation, evidence validation, prediction, explainability, and reporting.

Its key responsibilities include:

1. Loading and preparing student data.
2. Building the required day-60 features.
3. Checking evidence availability and routing ineligible cases.
4. Calling the prediction API for eligible records.
5. Requesting SHAP and LIME explanations.
6. Combining results for reporting and human assessment.

### Verified workflow demonstration

A verified execution using real dataset records produced the following results:

| Workflow outcome | Count |
|---|---:|
| Records processed | 10 |
| Records assessed by the model | 6 |
| Records routed for human review | 4 |
| Execution errors | 0 |

The four review cases had been unregistered before the day-60 cutoff and were not scored.

A separate synthetic API test was used to verify prediction and explanation functionality. Synthetic test results are technical demonstrations and must not be represented as predictions for real students.

## 9. Technology Stack

- **Python** — Data processing and machine learning.
- **Pandas and scikit-learn** — Feature processing and model development.
- **Random Forest** — Academic outcome classification.
- **SHAP and LIME** — Model explainability.
- **FastAPI** — Prediction and explanation API.
- **n8n** — Workflow orchestration and automation.
- **Render** — API deployment.
- **OULAD** — Learning analytics dataset.

## 10. Getting Started

### Access the live system

The fastest way to explore the API is through the hosted documentation:

https://the-data-detective-idc3.onrender.com/docs

From there, you can inspect the available endpoints and review their expected request formats.

### Run the project locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Abasifreke18/the_data_detective_idc3.git
   cd the_data_detective_idc3
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python -m venv .venv
   ```

   On Windows:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the project's dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Follow the repository's application entry point and configuration to start the API locally. Ensure the trained model artifact and any required dataset files are available at the paths expected by the application.

**Note:** The local startup command and model-artifact path should match the actual files in the repository.

## 11. Responsible AI and Limitations

Data Detective is a decision-support prototype, not an automated system for making final academic decisions.

Important considerations include:

- Model predictions are estimates, not confirmed student outcomes.
- Missing or insufficient evidence should trigger further review.
- SHAP and LIME explanations do not establish causation.
- Evaluation metrics may not generalize to different student populations.
- Real-world deployment requires additional validation, privacy safeguards, fairness assessment, and educator feedback.
- Student data should be handled securely and accessed only for authorized purposes.

The system is intended to help educators identify potential concerns and investigate them, not to replace professional judgment or label students permanently.

## 12. Future Improvements

Potential next steps include:

- Testing the model on additional cohorts and independent datasets.
- Improving evidence validation and handling of incomplete records.
- Evaluating fairness and performance across student groups.
- Developing a clearer educator-facing dashboard.
- Gathering feedback from educators and students.
- Evaluating whether the system leads to timely and effective student support.

## 13. Hackathon Information

**Event:** Ibom Data Connect 3.0  
**Theme:** Beyond Automation — Data Powering AI Agents  
**Track:** EdTech  
**Project:** Data Detective — Evidence-First Student Early Warning System  
**Team:** Data Detective

## Conclusion

Data Detective demonstrates how machine learning, explainable AI, and workflow automation can work together to support earlier identification of potential academic risk.

By combining predictions with evidence validation, explanations, and human review, the project aims to make student early warning systems more transparent, responsible, and useful to educators.

**Data Detective — Evidence first. Explanations included. Human judgment retained.**
