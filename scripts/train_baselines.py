"""TF-IDF + Logistic Regression / XGBoost baselines for both heads.

Establishes the numeric floor with classical ML before any GPU model
touches this data -- whatever DistilBERT achieves later has to actually
beat these numbers, not just exist alongside them.

Reads from model_ready_data (narrative, product, priority_bucket, split
only -- see sql/008_model_ready_data.sql) and routes every row through
extract_model_features() before it touches a vectorizer, so the same
leakage-boundary contract tested in tests/test_leakage.py is the one
actually used here, not a parallel hand-rolled column selection.
"""

import os
import time

import mlflow
import pandas as pd
import psycopg
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from ticket_triage.features import extract_model_features

load_dotenv()
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ticket-triage-baselines")

HEADS = {
    "product": "product",
    "priority": "priority_bucket",
}


def load_split(split: str) -> pd.DataFrame:
    conn = psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    df = pd.read_sql(
        "SELECT consumer_complaint_narrative, product, priority_bucket FROM model_ready_data WHERE split = %(split)s",
        conn,
        params={"split": split},
    )
    conn.close()
    # Route every row through the same leakage-boundary contract the
    # pytest suite verifies -- narrative text is the only thing that
    # reaches a vectorizer, not a parallel hand-rolled column pick.
    narratives = [extract_model_features(row)["narrative"] for _, row in df.iterrows()]
    df["narrative"] = narratives
    return df


def evaluate_and_log(model, X_val, y_val_enc, label_encoder, head_name, model_name):
    y_pred_enc = model.predict(X_val)
    y_val = label_encoder.inverse_transform(y_val_enc)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    macro_f1 = f1_score(y_val, y_pred, average="macro")
    weighted_f1 = f1_score(y_val, y_pred, average="weighted")
    report = classification_report(y_val, y_pred, zero_division=0)

    mlflow.log_metric("macro_f1", macro_f1)
    mlflow.log_metric("weighted_f1", weighted_f1)
    mlflow.log_text(report, "classification_report.txt")

    fig_path = f"/tmp/{head_name}_{model_name}_confusion_matrix.png"
    disp = ConfusionMatrixDisplay.from_predictions(
        y_val, y_pred, xticks_rotation="vertical", labels=label_encoder.classes_
    )
    disp.figure_.set_size_inches(8, 8)
    disp.figure_.tight_layout()
    disp.figure_.savefig(fig_path, dpi=110)
    mlflow.log_artifact(fig_path, "confusion_matrix.png")

    print(f"\n=== {head_name} / {model_name} ===")
    print(f"macro F1: {macro_f1:.4f}   weighted F1: {weighted_f1:.4f}")
    print(report)
    return macro_f1, weighted_f1


def train_head(head_name, y_col, train_df, val_df, X_train_tfidf, X_val_tfidf):
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(train_df[y_col])
    y_val_enc = label_encoder.transform(val_df[y_col])
    n_classes = len(label_encoder.classes_)
    print(f"\n--- Head: {head_name} ({n_classes} classes) ---")

    with mlflow.start_run(run_name=f"{head_name}-logreg"):
        mlflow.log_params(
            {
                "head": head_name,
                "model": "logistic_regression",
                "class_weight": "balanced",
                "n_classes": n_classes,
                "n_train": len(train_df),
            }
        )
        start = time.time()
        clf = LogisticRegression(
            class_weight="balanced", max_iter=200, n_jobs=-1, solver="lbfgs"
        )
        clf.fit(X_train_tfidf, y_train_enc)
        train_time = time.time() - start
        mlflow.log_metric("train_seconds", train_time)
        print(f"logreg trained in {train_time:.0f}s")
        evaluate_and_log(clf, X_val_tfidf, y_val_enc, label_encoder, head_name, "logreg")

    with mlflow.start_run(run_name=f"{head_name}-xgboost"):
        mlflow.log_params(
            {
                "head": head_name,
                "model": "xgboost",
                "class_weight": "balanced (via sample_weight)",
                "n_classes": n_classes,
                "n_train": len(train_df),
            }
        )
        sample_weight = compute_sample_weight("balanced", y_train_enc)
        start = time.time()
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=200,
            max_depth=6,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        clf.fit(X_train_tfidf, y_train_enc, sample_weight=sample_weight)
        train_time = time.time() - start
        mlflow.log_metric("train_seconds", train_time)
        print(f"xgboost trained in {train_time:.0f}s")
        evaluate_and_log(clf, X_val_tfidf, y_val_enc, label_encoder, head_name, "xgboost")


def main():
    print("Loading train split...")
    train_df = load_split("train")
    print(f"  {len(train_df):,} rows")
    print("Loading val split...")
    val_df = load_split("val")
    print(f"  {len(val_df):,} rows")

    print("Fitting shared TF-IDF vectorizer on train narratives...")
    vectorizer = TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 2), min_df=5, stop_words="english"
    )
    start = time.time()
    X_train_tfidf = vectorizer.fit_transform(train_df["narrative"])
    X_val_tfidf = vectorizer.transform(val_df["narrative"])
    print(f"  vectorized in {time.time() - start:.0f}s, vocab size {len(vectorizer.vocabulary_):,}")

    for head_name, y_col in HEADS.items():
        train_head(head_name, y_col, train_df, val_df, X_train_tfidf, X_val_tfidf)

    print("\nDone. View results with: uv run mlflow ui")


if __name__ == "__main__":
    main()
