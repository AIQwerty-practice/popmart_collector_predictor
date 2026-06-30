from pathlib import Path
import sys

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"

sys.path.insert(0, str(PROJECT_ROOT))

from app_api.main import app  # noqa: E402


client = TestClient(app)


def sample_payload() -> dict:
    return {
        "age": 29,
        "monthly_budget_usd": 120,
        "collection_size": 36,
        "monthly_purchases": 5,
        "resale_interest": 4,
        "social_media_engagement": 8,
        "blind_box_risk_tolerance": 7,
        "favorite_series": "Labubu",
        "collector_type": "completionist",
        "region": "North America",
    }


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "POP MART Collector Predictor"


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_available"] == MODEL_PATH.exists()


def test_predict() -> None:
    assert MODEL_PATH.exists(), "Run python scripts/train_model.py before API tests."

    response = client.post("/predict", json=sample_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["label"] in ["High collector engagement", "Lower collector engagement"]
    assert 0 <= body["probability"] <= 1
    assert body["interpretation"]


def test_predict_validation() -> None:
    payload = sample_payload()
    payload["age"] = 12

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
