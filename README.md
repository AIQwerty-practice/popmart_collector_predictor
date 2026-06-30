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

POP MART Collector Predictor is an end-to-end MLOps mini project that predicts a POP MART collector profile/level:

- Casual
- Enthusiast
- Hardcore

The project uses synthetic collector behavior data, a trained Random Forest pipeline, a FastAPI model-serving backend, and a Streamlit companion dashboard.

## Final Architecture

```text
Streamlit frontend
-> sends JSON payload to deployed FastAPI backend
-> FastAPI hosted on Hugging Face Spaces
-> FastAPI loads model/model.pkl
-> Random Forest pipeline predicts collector level
-> API returns prediction and probability
-> Streamlit displays result card, probability bars, and Mystery Box Insight
```

In the final version, Streamlit acts as the frontend and sends user input to the deployed FastAPI service. FastAPI handles model serving and returns a JSON response. Streamlit then turns the response into a presentation-friendly collector profile card.

## Dataset And Model

The project uses exactly 2,000 synthetic POP MART collector records.

The training workflow uses:

- `scikit-learn Pipeline`
- `ColumnTransformer`
- `OneHotEncoder`
- `RandomForestClassifier`

The trained pipeline is saved at:

```text
model/model.pkl
```

## FastAPI Backend

FastAPI provides the model-serving layer.

Endpoints:

```text
GET /
GET /health
POST /predict
```

Live prediction endpoint:

```text
https://bee-ai-popmart-collector-predictor.hf.space/predict
```

Interactive API documentation:

```text
https://bee-ai-popmart-collector-predictor.hf.space/docs
```

Health check:

```text
https://bee-ai-popmart-collector-predictor.hf.space/health
```

## Example API Usage

The `/predict` endpoint accepts a JSON collector profile with numeric collecting habits, budget information, interest scores, and categorical profile choices. The exact request schema is defined in `app_api/main.py` and is visible in the deployed `/docs` page.

Example request:

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

The API returns the model prediction and model probability. Streamlit uses that probability to display Casual, Enthusiast, and Hardcore probability bars.

## Streamlit Frontend

The Streamlit app is the user-facing dashboard. It collects user inputs, sends a JSON request to the deployed FastAPI backend, receives the prediction response, and displays:

- Collector profile result card
- Probability bars
- Character-themed visual elements
- Mystery Box Insight
- Educational model disclaimer

## Deployment

The FastAPI backend is deployed on Hugging Face Spaces using Docker.

The `Dockerfile` is used to:

- Install the Python dependencies.
- Copy the application files into the container.
- Start FastAPI with `uvicorn`.
- Expose port `7860`, which Hugging Face Spaces expects for Docker apps.

Docker start command:

```dockerfile
CMD ["python", "-m", "uvicorn", "app_api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```


## Run Locally

Install dependencies:

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

Run FastAPI:

```powershell
uvicorn app_api.main:app --reload
```

Run Streamlit:

```powershell
streamlit run app_streamlit/app.py
```

Run tests:

```powershell
pytest tests/test_api.py
```

If command shortcuts are not available, use Python module form:

```powershell
python -m uvicorn app_api.main:app --reload
python -m streamlit run app_streamlit/app.py
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
Stores the generated synthetic dataset.

```text
scripts/train_model.py
```
Trains the scikit-learn pipeline and saves the trained model artifact.

```text
model/model.pkl
```
Stores the trained Random Forest pipeline used by FastAPI.

```text
app_api/main.py
```
Defines the FastAPI backend, API routes, request validation, model loading, and prediction response.

```text
app_streamlit/app.py
```
Defines the Streamlit dashboard that calls the deployed FastAPI backend and displays the collector profile result.

```text
tests/test_api.py
```
Contains automated tests for API availability, prediction behavior, and request validation.

```text
requirements.txt
```
Lists Python dependencies for data generation, model training, API serving, dashboard UI, and testing.

```text
Dockerfile
```
Defines how Hugging Face Spaces builds and runs the FastAPI backend.

```text
README.md
```
Documents the project, run instructions, architecture, deployment, and Hugging Face Space metadata.

```text
.gitignore
```
Excludes virtual environments, cache files, and compiled Python files from version control.

## Model Disclaimer

This is an educational mini project using synthetic data. Predictions are for demonstration purposes only.
