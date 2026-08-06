"""LogReg reruns for the evaluation report: raw predictions + a binary
Priority variant.

Two things the original baseline run (scripts/train_baselines.py) didn't
produce, both needed for the evaluation report:

1. Raw per-row predictions with narrative text and complaint_id, so
   specific misclassifications can actually be read and discussed in the
   qualitative error analysis -- the original run only logged aggregate
   metrics and a confusion matrix image, not the underlying predictions.
2. A LogReg trained directly on a binary Priority target (Low vs
   Elevated = Medium-or-High) -- the build plan's own named fallback if
   3-way Priority proves unworkable, which the baseline/DistilBERT/
   zero-shot comparison now gives real evidence for. This is a model
   trained ON the binary target, not a post-hoc relabeling of 3-way
   predictions -- a real, if small, difference in what the model can
   learn from the training signal.

Only LogReg is rerun here (13-37s per head originally), not XGBoost
(28 minutes for Product) -- LogReg already represents the "trained
model's conservative failure mode" on Priority just as clearly, and
the marginal analytical value of also rerunning the slow model isn't
worth 28+ more minutes.
"""

import os
import time
from pathlib import Path

import mlflow
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from ticket_triage.features import extract_model_features

load_dotenv()
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ticket-triage-baselines")


def load_split(split: str) -> pd.DataFrame:
    conn = psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    df = pd.read_sql(
        "SELECT complaint_id, consumer_complaint_narrative, product, priority_bucket "
        "FROM model_ready_data WHERE split = %(split)s",
        conn,
        params={"split": split},
    )
    conn.close()
    df["narrative"] = [extract_model_features(row)["narrative"] for _, row in df.iterrows()]
    return df


def fit_and_predict(head_name, y_col, train_df, val_df, X_train_tfidf, X_val_tfidf, run_suffix=""):
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(train_df[y_col])
    y_val_enc = label_encoder.transform(val_df[y_col])
    n_classes = len(label_encoder.classes_)
    print(f"\n--- {head_name}{run_suffix}: {n_classes} classes {list(label_encoder.classes_)} ---")

    with mlflow.start_run(run_name=f"{head_name}{run_suffix}-logreg-erranalysis"):
        mlflow.log_params(
            {"head": head_name, "model": "logistic_regression", "class_weight": "balanced", "n_classes": n_classes}
        )
        start = time.time()
        clf = LogisticRegression(class_weight="balanced", max_iter=200, solver="lbfgs")
        clf.fit(X_train_tfidf, y_train_enc)
        train_time = time.time() - start
        print(f"trained in {train_time:.0f}s")

        y_pred_enc = clf.predict(X_val_tfidf)
        y_val = label_encoder.inverse_transform(y_val_enc)
        y_pred = label_encoder.inverse_transform(y_pred_enc)

        macro_f1 = f1_score(y_val, y_pred, average="macro")
        weighted_f1 = f1_score(y_val, y_pred, average="weighted")
        report = classification_report(y_val, y_pred, zero_division=0)
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("weighted_f1", weighted_f1)
        mlflow.log_text(report, "classification_report.txt")
        print(f"macro F1: {macro_f1:.4f}   weighted F1: {weighted_f1:.4f}")
        print(report)

    result_df = val_df[["complaint_id", "narrative", y_col]].copy()
    result_df["pred"] = y_pred
    return result_df, macro_f1, weighted_f1


def main():
    print("Loading train/val splits...")
    train_df = load_split("train")
    val_df = load_split("val")
    print(f"  train {len(train_df):,}, val {len(val_df):,}")

    print("Fitting shared TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=5, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(train_df["narrative"])
    X_val_tfidf = vectorizer.transform(val_df["narrative"])

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Product, 3-way Priority -- raw predictions for error analysis
    product_preds, _, _ = fit_and_predict("product", "product", train_df, val_df, X_train_tfidf, X_val_tfidf)
    product_preds.to_parquet(out_dir / "logreg_product_val_predictions.parquet", index=False)

    priority_preds, _, _ = fit_and_predict(
        "priority", "priority_bucket", train_df, val_df, X_train_tfidf, X_val_tfidf
    )
    priority_preds.to_parquet(out_dir / "logreg_priority_val_predictions.parquet", index=False)

    # 2. Binary Priority: Low vs Elevated (Medium-or-High)
    train_df["priority_binary"] = train_df["priority_bucket"].apply(
        lambda b: "Low" if b == "Low" else "Elevated"
    )
    val_df["priority_binary"] = val_df["priority_bucket"].apply(
        lambda b: "Low" if b == "Low" else "Elevated"
    )
    binary_preds, binary_macro_f1, binary_weighted_f1 = fit_and_predict(
        "priority", "priority_binary", train_df, val_df, X_train_tfidf, X_val_tfidf, run_suffix="-binary"
    )
    binary_preds.to_parquet(out_dir / "logreg_priority_binary_val_predictions.parquet", index=False)

    print("\nDone.")
    print(f"Binary Priority (Low vs Elevated) macro F1: {binary_macro_f1:.4f}, weighted F1: {binary_weighted_f1:.4f}")


if __name__ == "__main__":
    main()
