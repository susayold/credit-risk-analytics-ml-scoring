# Model Evidence Pack

This folder contains model cards and artifact metadata for the credit risk project.

The project is an analytics and ML scoring support case study. It does not ship a binary production model object by default. Instead, it commits the evidence needed to audit model selection and business use:

- champion model card
- diagnostic Logistic Regression model card
- metric source-of-truth mapping
- output tables with model metrics, SHAP, PSI, fairness, and decision bands

## Files

| File | Purpose |
|---|---|
| `champion_model_card.json` | Machine-readable summary of the LightGBM v3 champion and calibrated ensemble challenger |
| `diagnostic_logit_model_card.json` | Machine-readable summary of the WOE-style Logistic Regression diagnostic benchmark |
| `model_artifact_manifest.csv` | Paths to the committed evidence files used as model artifacts |

## Why No Binary Model Object

The project is positioned for portfolio review and business analytics. The committed artifacts are enough to explain how the model was selected and evaluated. A production implementation would add a serialized model, encoder/preprocessor object, feature schema lock, monitoring jobs, approval policy, compliance controls, and a serving API.
