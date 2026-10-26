import pytest
from fastapi.testclient import TestClient

from src.app.main import app, model

client = TestClient(app)

HIGH_RISK = {
    "tenure_months": 2,
    "monthly_fee": 95.5,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "support_calls": 5,
}


def test_health_contract():
    """API contract: /health always answers 200 with status ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_contract():
    """API contract: /predict returns churn (0/1) and a probability in [0, 1]."""
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn", "probability"}
    assert body["churn"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_model_loads():
    """The saved model file loads and can produce probabilities."""
    assert hasattr(model, "predict_proba")


def test_unknown_category_rejected():
    """Input schema: a category the model never saw is rejected with 422."""
    r = client.post("/predict", json={**HIGH_RISK, "internet_service": "cable"})
    assert r.status_code == 422


def test_missing_field_rejected():
    """Input schema: a missing field is rejected with 422."""
    bad = {k: v for k, v in HIGH_RISK.items() if k != "support_calls"}
    assert client.post("/predict", json=bad).status_code == 422


@pytest.mark.parametrize("tenure, expected_status", [(0, 200), (-1, 422)])
def test_tenure_edge_cases(tenure, expected_status):
    """Edge case: a brand-new customer (tenure 0) is valid; negative tenure is not."""
    r = client.post("/predict", json={**HIGH_RISK, "tenure_months": tenure})
    assert r.status_code == expected_status


def test_deterministic_prediction():
    """Deterministic: the same input always gives the same, known answer."""
    first = client.post("/predict", json=HIGH_RISK).json()
    second = client.post("/predict", json=HIGH_RISK).json()
    assert first == second
    assert first["churn"] == 1
    assert first["probability"] == pytest.approx(0.9532, abs=1e-3)