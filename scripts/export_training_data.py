"""Export model_ready_data to a single Parquet file for Colab.

Colab has no way to reach this machine's local Postgres container, so
the data has to be exported to a portable file, uploaded to Google
Drive, and read from there in the fine-tuning notebook. This script is
the local-side half of that handoff.

Every row is routed through extract_model_features() before export --
the same leakage-boundary contract used everywhere else in this
project -- so the exported file physically cannot carry an outcome
column, regardless of what model_ready_data happens to expose.
"""

import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

from ticket_triage.features import extract_model_features

load_dotenv()

OUT_PATH = Path("data/processed/model_ready_data.parquet")


def main():
    conn = psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    print("Querying model_ready_data...")
    df = pd.read_sql(
        "SELECT complaint_id, consumer_complaint_narrative, product, priority_bucket, split FROM model_ready_data",
        conn,
    )
    conn.close()
    print(f"  {len(df):,} rows")

    print("Routing through extract_model_features()...")
    df["narrative"] = [extract_model_features(r)["narrative"] for _, r in df.iterrows()]
    # complaint_id is carried as a row identifier for later error analysis
    # (tracing a specific misclassification back to its source row) -- it
    # is never used as a model input.
    export_df = df[["complaint_id", "narrative", "product", "priority_bucket", "split"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT_PATH} ({size_mb:.1f} MB)")

    print("\nSplit sizes:")
    print(export_df["split"].value_counts())


if __name__ == "__main__":
    main()
