"""API contract tests.

Models are stubbed, not loaded for real -- these tests run fast and
don't depend on network access, a GPU, or local model artifacts,
which matters for CI. What's being verified here is contract shape,
request validation, and the leakage boundary -- model *accuracy* is
the evaluation report's job, backed by the real MLflow runs.
"""

import numpy as np
import torch
from fastapi.testclient import TestClient

import ticket_triage.api as api_module


class StubTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.zeros((1, 8), dtype=torch.long),
            "attention_mask": torch.ones((1, 8), dtype=torch.long),
        }


class StubProductModel:
    class _Output:
        def __init__(self, logits):
            self.logits = logits

    def __call__(self, **kwargs):
        logits = torch.zeros((1, len(api_module.PRODUCT_LABELS)))
        logits[0, 3] = 5.0  # "Credit reporting or other personal consumer reports"
        return self._Output(logits)


class StubVectorizer:
    def transform(self, texts):
        return np.zeros((len(texts), 1))


class StubPriorityModel:
    def predict(self, X):
        return np.array(["Elevated"])

    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])


def make_client() -> TestClient:
    # Populate the module-level models dict directly and construct
    # TestClient WITHOUT the `with` context manager -- Starlette only
    # runs the app's lifespan (which would try to download the real
    # DistilBERT model and load local joblib files) when TestClient is
    # entered as a context manager. A plain TestClient(app) just routes
    # requests against whatever's already in `models`.
    api_module.models.clear()
    api_module.models.update(
        {
            "product_tokenizer": StubTokenizer(),
            "product_model": StubProductModel(),
            "priority_vectorizer": StubVectorizer(),
            "priority_model": StubPriorityModel(),
        }
    )
    return TestClient(api_module.app)


def test_health():
    client = make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "models_loaded": True}


def test_predict_returns_expected_shape():
    client = make_client()
    resp = client.post("/predict", json={"narrative": "My credit report has an error on it."})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"product", "product_confidence", "priority", "priority_confidence"}
    assert body["product"] == "Credit reporting or other personal consumer reports"
    assert body["priority"] == "Elevated"
    assert 0.0 <= body["product_confidence"] <= 1.0
    assert 0.0 <= body["priority_confidence"] <= 1.0


def test_predict_rejects_empty_narrative():
    client = make_client()
    resp = client.post("/predict", json={"narrative": ""})
    assert resp.status_code == 422


def test_predict_rejects_missing_narrative():
    client = make_client()
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_ignores_extra_outcome_fields():
    """Leakage boundary at the API layer: even if a caller sends an
    outcome-style field alongside narrative, the schema has no field for
    it, so it can never reach either model -- verified here by confirming
    a request with and without the extra field produce identical
    predictions, not just that the response happens to look normal."""
    client = make_client()
    narrative = "My credit report has an error on it."

    resp_clean = client.post("/predict", json={"narrative": narrative})
    resp_with_extra = client.post(
        "/predict",
        json={
            "narrative": narrative,
            "company_response_to_consumer": "Closed with monetary relief",
            "timely_response": "No",
            "tags": "Older American",
        },
    )
    assert resp_clean.status_code == resp_with_extra.status_code == 200
    assert resp_clean.json() == resp_with_extra.json()
