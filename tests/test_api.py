import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_predict_endpoint_valid_payload():
    payload = {
        "features": [0.0] * 187
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "latency_seconds" in data

def test_predict_endpoint_invalid_payload_length():
    payload = {
        "features": [0.0] * 100
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Invalid input length" in response.json()["detail"]

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"api_predict_requests_total" in response.content
