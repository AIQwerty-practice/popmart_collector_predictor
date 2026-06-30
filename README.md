---
title: POPMart Collector Predictor
emoji: 🎁
colorFrom: pink
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# POP MART Collector Predictor

POP MART Collector Predictor is an end-to-end MLOps mini project that estimates a collector profile from collecting habits. The project combines synthetic data generation, model training, FastAPI model serving, a Streamlit frontend, automated tests, Docker deployment, GitHub version control, and Hugging Face Spaces hosting.

The final architecture is:

```text
Streamlit frontend -> Hugging Face Spaces FastAPI backend -> Random Forest pipeline -> JSON prediction response -> Streamlit result dashboard
```

## Project Overview

This project uses a reproducible synthetic dataset of exactly 2,000 POP MART collector records. Each record includes collector attributes such as age, monthly budget, collection size, monthly purchases, resale interest, social media engagement, blind box risk tolerance, favorite series, collector type, and region.

The machine learning model is a scikit-learn `Pipeline` that uses:

- `ColumnTransformer`
- `OneHotEncoder`
- `RandomForestClassifier`

The trained pipeline is saved as:

```text
model/model.pkl
```

FastAPI serves the model through REST endpoints, and Streamlit sends user input to the deployed FastAPI backend for prediction. Streamlit then displays the collector result, probability bars, character styling, and a fun Mystery Box Insight.

## Final Architecture

```text
User input in Streamlit
-> Streamlit sends JSON payload
-> Hugging Face Spaces FastAPI /predict endpoint
-> Random Forest model pipeline
-> API returns JSON prediction and probability
-> Streamlit displays collector type, probability bars, result card, and Mystery Box Insight
```

## FastAPI Endpoints

The backend is implemented in `app_api/main.py`.

Available endpoints:

```text
GET /
GET /health
POST /predict
```

The deployed API is hosted on Hugging Face Spaces using Docker.

Live prediction endpoint:

```text
https://bee-ai-popmart-collector-predictor.hf.space/predict
```

Interactive API docs:

```text
https://bee-ai-popmart-collector-predictor.hf.space/docs
```

Health check:

```text
https://bee-ai-popmart-collector-predictor.hf.space/health
```

## Example API Request

`POST /predict`

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

Example response:

```json
{
  "prediction": 1,
  "label": "Will buy next release",
  "probability": 0.8125,
  "interpretation": "Very likely to buy the next release."
}
```

Note: the API returns the model's positive-class probability. The Streamlit dashboard uses that probability to display collector levels and visual probability bars for Casual, Enthusiast, and Hardcore collector profiles.

## Deployment

The FastAPI backend is deployed to Hugging Face Spaces as a Docker Space.

The `Dockerfile`:

- Uses a lightweight Python image.
- Installs dependencies from `requirements.txt`.
- Copies the project files into the container.
- Starts FastAPI with `uvicorn`.
- Exposes the Hugging Face-required port `7860`.

Docker start command:

```dockerfile
CMD ["python", "-m", "uvicorn", "app_api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

GitHub is used for source control and project versioning.

## Run Locally

From the project root:

```powershell
python -m pip install -r requirements.txt
```

Generate the synthetic dataset:

```powershell
python scripts/generate_dataset.py
```

Train the model:

```powershell
python scripts/train_model.py
```

Run FastAPI locally:

```powershell
uvicorn app_api.main:app --reload
```

If `uvicorn` is not on PATH, use:

```powershell
python -m uvicorn app_api.main:app --reload
```

Run Streamlit:

```powershell
streamlit run app_streamlit/app.py
```

If `streamlit` is not on PATH, use:

```powershell
python -m streamlit run app_streamlit/app.py
```

By default, the Streamlit app calls the deployed Hugging Face FastAPI backend:

```text
https://bee-ai-popmart-collector-predictor.hf.space
```

To point Streamlit to a local FastAPI backend instead:

```powershell
$env:FASTAPI_URL="http://127.0.0.1:8000"
python -m streamlit run app_streamlit/app.py
```

## Test

Run API tests:

```powershell
pytest tests/test_api.py
```

If `pytest` is not on PATH, use:

```powershell
python -m pytest tests/test_api.py
```

## Project Structure

```text
scripts/generate_dataset.py
```
Generates the reproducible synthetic dataset of 2,000 POP MART collector records.

```text
data/popmart_collectors.csv
```
Stores the generated synthetic collector dataset.

```text
scripts/train_model.py
```
Trains the scikit-learn Pipeline with `OneHotEncoder`, `ColumnTransformer`, and `RandomForestClassifier`.

```text
model/model.pkl
```
Stores the trained machine learning pipeline used by FastAPI.

```text
app_api/main.py
```
Defines the FastAPI model-serving backend with `/`, `/health`, and `/predict`.

```text
app_streamlit/app.py
```
Defines the Streamlit dashboard. It sends prediction requests to the deployed FastAPI backend and displays collector profile results.

```text
tests/test_api.py
```
Contains automated tests for FastAPI root, health, prediction, and validation behavior.

```text
requirements.txt
```
Lists Python dependencies needed for data generation, training, API serving, Streamlit, and testing.

```text
Dockerfile
```
Builds and runs the FastAPI backend on Hugging Face Spaces.

```text
README.md
```
Documents the project, architecture, usage, deployment, and Hugging Face Space metadata.

```text
.gitignore
```
Prevents local virtual environments, caches, and compiled Python files from being committed.

## Model Disclaimer

This is an educational mini project using synthetic data. Predictions are for demonstration purposes only.

## Suggested GitHub Description

```text
End-to-end MLOps mini project with FastAPI, Streamlit, Random Forest, Docker, and Hugging Face Spaces deployment.
```
