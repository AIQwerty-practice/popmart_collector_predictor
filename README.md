# POP MART Collector Predictor

Predict whether a synthetic POP MART collector is likely to buy the next release.

The project includes:

- A reproducible dataset generator for exactly 2,000 collectors.
- A scikit-learn `Pipeline` using `ColumnTransformer`, `OneHotEncoder`, and `RandomForestClassifier`.
- A FastAPI app with `/`, `/health`, and `/predict`.
- A Streamlit interface with profile controls, probability output, and interpretation.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Generate Data

```powershell
python scripts/generate_dataset.py
```

This writes:

```text
data/popmart_collectors.csv
```

## Train Model

```powershell
python scripts/train_model.py
```

This writes:

```text
model/model.pkl
```

## Run API

```powershell
uvicorn app_api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Example prediction payload:

```json
{
  "age": 29,
  "monthly_budget_usd": 120,
  "collection_size": 36,
  "monthly_purchases": 5,
  "resale_interest": 4,
  "social_media_engagement": 8,
  "blind_box_risk_tolerance": 7,
  "favorite_series": "Labubu",
  "collector_type": "completionist",
  "region": "North America"
}
```

## Run Streamlit

```powershell
streamlit run app_streamlit/app.py
```

## Test

```powershell
pytest tests/test_api.py
```
