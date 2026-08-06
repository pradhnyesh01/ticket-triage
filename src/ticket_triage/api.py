"""FastAPI serving app for the ticket triage classifier.

Two models, deliberately different architectures, chosen directly from
the evaluation results (see README.md's Results section):
- Product: fine-tuned DistilBERT (0.614 macro F1, best of the four
  approaches tested).
- Priority: binary Logistic Regression, Low vs Elevated (0.605 macro
  F1, 75% recall) -- not any of the 3-way Low/Medium/High models, all
  three of which have 0% recall on High and would be indefensible to
  ship after finding and explaining exactly why they fail.

PredictRequest accepts only narrative text (see schemas.py) -- the
leakage boundary is enforced structurally here too, not just by
convention: there is no field for an outcome column to occupy.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import torch
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ticket_triage.schemas import PredictRequest, PredictResponse

PRODUCT_MODEL_REPO = os.environ.get("PRODUCT_MODEL_REPO", "praddyvk/ticket-triage-product-distilbert")
PRIORITY_MODEL_DIR = Path(os.environ.get("PRIORITY_MODEL_DIR", "models/priority"))
MAX_LENGTH = 256

# The DistilBERT model on HF Hub carries no label metadata -- the Colab
# notebook tracked the index<->category mapping only in a local, never-
# uploaded sklearn LabelEncoder. LabelEncoder.fit_transform() sorts
# classes via np.unique(), i.e. plain alphabetical order for strings --
# deterministic and reproducible, so this list (verified against the
# notebook's own printed class order during training) reconstructs the
# exact same mapping. Verified again below by loading the real model and
# checking predictions against obviously-labeled examples before this
# was trusted.
PRODUCT_LABELS = [
    "Checking or savings account",
    "Credit card",
    "Credit card or prepaid card",
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Debt or credit management",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Other financial service",
    "Payday loan, title loan, personal loan, or advance loan",
    "Prepaid card",
    "Student loan",
    "Vehicle loan or lease",
]

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading Product model from {PRODUCT_MODEL_REPO}...")
    models["product_tokenizer"] = AutoTokenizer.from_pretrained(PRODUCT_MODEL_REPO)
    models["product_model"] = AutoModelForSequenceClassification.from_pretrained(PRODUCT_MODEL_REPO)
    models["product_model"].eval()

    print(f"Loading Priority model from {PRIORITY_MODEL_DIR}...")
    models["priority_vectorizer"] = joblib.load(PRIORITY_MODEL_DIR / "tfidf_vectorizer.joblib")
    models["priority_model"] = joblib.load(PRIORITY_MODEL_DIR / "logreg_binary.joblib")

    print("Models loaded.")
    yield
    models.clear()


app = FastAPI(
    title="Support Ticket Triage & Priority Classifier",
    description="Predicts Product category and Priority (Low/Elevated) from a consumer complaint narrative alone.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(models) > 0}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    narrative = request.narrative

    tokenizer = models["product_tokenizer"]
    product_model = models["product_model"]
    inputs = tokenizer(
        narrative, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        logits = product_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    product_idx = int(torch.argmax(probs))
    product_label = PRODUCT_LABELS[product_idx]
    product_confidence = float(probs[product_idx])

    vectorizer = models["priority_vectorizer"]
    priority_model = models["priority_model"]
    X = vectorizer.transform([narrative])
    priority_label = priority_model.predict(X)[0]
    priority_confidence = float(max(priority_model.predict_proba(X)[0]))

    return PredictResponse(
        product=product_label,
        product_confidence=product_confidence,
        priority=priority_label,
        priority_confidence=priority_confidence,
    )
