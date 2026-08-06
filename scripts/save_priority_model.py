"""Train and persist the binary Priority model (Low vs Elevated) for serving.

This is the model that actually gets deployed for the Priority head --
not any of the 3-way models, all of which have 0% recall on High and
would be indefensible to ship. See README.md's Results section for why.

Model files are saved to models/priority/ as real build artifacts
(gitignored, like checkpoints/ and mlruns/ -- regenerate by rerunning
this script, don't hand-edit or expect them in git). The serving app
loads directly from these files, with no Postgres dependency at runtime
-- a deployed service should never need a connection back to the local
dev database.
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from ticket_triage.features import extract_model_features

load_dotenv()

OUT_DIR = Path("models/priority")


def load_split(split: str) -> pd.DataFrame:
    conn = psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    df = pd.read_sql(
        "SELECT consumer_complaint_narrative, priority_bucket FROM model_ready_data WHERE split = %(split)s",
        conn,
        params={"split": split},
    )
    conn.close()
    df["narrative"] = [extract_model_features(row)["narrative"] for _, row in df.iterrows()]
    df["priority_binary"] = df["priority_bucket"].apply(lambda b: "Low" if b == "Low" else "Elevated")
    return df


def main():
    print("Loading train/val splits...")
    train_df = load_split("train")
    val_df = load_split("val")

    print("Fitting TF-IDF vectorizer + binary Logistic Regression...")
    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=5, stop_words="english")
    X_train = vectorizer.fit_transform(train_df["narrative"])
    X_val = vectorizer.transform(val_df["narrative"])

    clf = LogisticRegression(class_weight="balanced", max_iter=200, solver="lbfgs")
    clf.fit(X_train, train_df["priority_binary"])

    y_pred = clf.predict(X_val)
    macro_f1 = f1_score(val_df["priority_binary"], y_pred, average="macro")
    print(f"val macro F1: {macro_f1:.4f}")
    print(classification_report(val_df["priority_binary"], y_pred, zero_division=0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, OUT_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(clf, OUT_DIR / "logreg_binary.joblib")
    meta = {
        "model": "logistic_regression",
        "target": "priority_binary (Low vs Elevated)",
        "classes": clf.classes_.tolist(),
        "val_macro_f1": macro_f1,
        "vectorizer_max_features": 50_000,
        "vectorizer_ngram_range": [1, 2],
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model, vectorizer, and metadata to {OUT_DIR}/")


if __name__ == "__main__":
    main()
