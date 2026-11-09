import pytest
from fastapi.testclient import TestClient

from app import app

GOOD = dict(amount=42.5, hour=14, distance_km=3.2, merchant="grocery", online=False, tx_last_24h=1)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # the `with` runs the startup: the model gets loaded
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_a_probability(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert 0 <= r.json()["fraud_proba"] <= 1


def test_negative_amount_is_rejected(client):
    r = client.post("/predict", json={**GOOD, "amount": -5})
    assert r.status_code == 422


def test_unknown_merchant_is_rejected(client):
    r = client.post("/predict", json={**GOOD, "merchant": "casino"})
    assert r.status_code == 422


def test_batch_gives_same_answer_as_single(client):
    single = client.post("/predict", json=GOOD).json()["fraud_proba"]
    batch = client.post("/predict_batch", json={"transactions": [GOOD, GOOD]}).json()["fraud_proba"]
    assert batch == pytest.approx([single, single])
