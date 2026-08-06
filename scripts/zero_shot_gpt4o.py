"""Zero-shot GPT-4o comparison for both heads, on the held-out test set.

Samples all 32 High-priority test rows (guaranteed full coverage of the
rare class both the TF-IDF baselines and fine-tuned DistilBERT failed on
completely -- 0/34 on the val set) plus a proportional stratified sample
of ~1,500 Medium/Low test rows. Prompts GPT-4o once per row for BOTH
Product and Priority in a single combined call -- the natural,
cost-efficient zero-shot pattern, deliberately different from the "two
separate single-task models" approach used for the trained baselines and
DistilBERT (that separation existed for training/interpretability
reasons that don't apply to a single stateless LLM call).

Only narrative text is ever sent to the API -- same leakage boundary as
everywhere else in this project. Latency and cost are computed from each
response's actual measured wall-clock time and reported token usage, not
estimated in advance.

IMPORTANT: metrics on this sample are NOT directly apples-to-apples with
the baseline/fine-tuned test-set metrics for weighted F1 or accuracy,
because High is deliberately massively over-represented here (32 of
~1532 rows, vs 32 of 37,499 in the real test set) so it can be examined
in detail. macro F1 and per-class precision/recall/F1 ARE valid for
comparison, since they don't depend on the sample's class proportions.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

load_dotenv()
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ticket-triage-zero-shot")

MODEL = "gpt-4o"
SEED = 42
N_SAMPLE_NON_HIGH = 1500
MAX_WORKERS = 8

# Approximate GPT-4o pricing as of this writing -- verify against
# https://openai.com/api/pricing/ before trusting the dollar figures,
# prices change. Actual token counts used to compute cost ARE measured
# from the API's real responses, only the per-token rate is a constant.
PRICE_PER_MTOK_INPUT = 2.50
PRICE_PER_MTOK_OUTPUT = 10.00

PRODUCTS = [
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Checking or savings account",
    "Mortgage",
    "Credit card",
    "Money transfer, virtual currency, or money service",
    "Credit card or prepaid card",
    "Student loan",
    "Vehicle loan or lease",
    "Payday loan, title loan, personal loan, or advance loan",
    "Prepaid card",
    "Debt or credit management",
    "Other financial service",
]

SYSTEM_PROMPT = f"""You are classifying US consumer complaints filed with the CFPB (Consumer Financial Protection Bureau). For each complaint, you are given ONLY the consumer's narrative description -- nothing else about how the complaint was handled, resolved, or by which company.

Classify each complaint into exactly two fields:

1. "product": the single best-matching category from this exact list (use the exact text):
{chr(10).join(f'- {p}' for p in PRODUCTS)}

2. "priority": one of "Low", "Medium", or "High", estimating how urgent/severe this complaint is likely to be. As general guidance (not information about this specific complaint): historically, complaints that end up receiving monetary relief, an untimely company response, involve an older American or servicemember, or concern high-severity issues (fraud, identity theft, threats of illegal action, debt collection for debt not owed, or wrongful credit denial) tend to be judged higher priority. You do not know the actual outcome of this complaint -- estimate from the narrative's content and tone alone.

Respond with ONLY a JSON object: {{"product": "...", "priority": "..."}}"""


def build_sample() -> pd.DataFrame:
    df = pd.read_parquet("data/processed/model_ready_data.parquet")
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    high_df = test_df[test_df["priority_bucket"] == "High"]
    rest_df = test_df[test_df["priority_bucket"] != "High"]

    parts = [high_df]
    for _, group in rest_df.groupby("priority_bucket"):
        n = round(N_SAMPLE_NON_HIGH * len(group) / len(rest_df))
        parts.append(group.sample(n=n, random_state=SEED))

    sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return sample


def classify_row(client: OpenAI, narrative: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            start = time.time()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": narrative[:6000]},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            latency = time.time() - start
            content = json.loads(resp.choices[0].message.content)
            return {
                "product_pred": content.get("product"),
                "priority_pred": content.get("priority"),
                "latency_s": latency,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "error": None,
            }
        except Exception as e:
            if attempt == retries - 1:
                return {
                    "product_pred": None,
                    "priority_pred": None,
                    "latency_s": None,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "error": str(e),
                }
            time.sleep(2**attempt)


def run_predictions(sample: pd.DataFrame) -> pd.DataFrame:
    client = OpenAI()
    results = [None] * len(sample)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(classify_row, client, row.narrative): i
            for i, row in enumerate(sample.itertuples())
        }
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(sample)} complete")
    return pd.concat([sample.reset_index(drop=True), pd.DataFrame(results)], axis=1)


def evaluate_head(df: pd.DataFrame, head_name: str, true_col: str, pred_col: str):
    valid = df[df[pred_col].notna()]
    n_failed = len(df) - len(valid)
    if n_failed:
        print(f"  {n_failed} rows failed to get a valid prediction (excluded from metrics)")

    y_true = valid[true_col]
    y_pred = valid[pred_col]

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, zero_division=0)

    print(f"\n=== {head_name}: zero-shot GPT-4o ===")
    print(f"macro F1: {macro_f1:.4f}   weighted F1 (see caveat -- sample is not population-proportional): {weighted_f1:.4f}")
    print(report)

    mlflow.log_metric(f"{head_name}_macro_f1", macro_f1)
    mlflow.log_metric(f"{head_name}_weighted_f1", weighted_f1)
    mlflow.log_text(report, f"{head_name}_classification_report.txt")

    labels = sorted(set(y_true) | set(y_pred))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, xticks_rotation="vertical", labels=labels
    )
    disp.figure_.set_size_inches(8, 8)
    disp.figure_.tight_layout()
    fig_path = f"/tmp/zero_shot_{head_name}_confusion_matrix.png"
    disp.figure_.savefig(fig_path, dpi=110)
    mlflow.log_artifact(fig_path, f"{head_name}_confusion_matrix.png")

    return macro_f1, weighted_f1


def main():
    print("Building evaluation sample...")
    sample = build_sample()
    n_high = (sample["priority_bucket"] == "High").sum()
    print(f"  {len(sample)} rows ({n_high} High, {len(sample) - n_high} Medium/Low)")

    with mlflow.start_run(run_name="zero-shot-gpt4o"):
        mlflow.log_params(
            {
                "model": MODEL,
                "n_sample": len(sample),
                "n_high": int(n_high),
                "temperature": 0,
            }
        )

        print("Calling GPT-4o...")
        start = time.time()
        results_df = run_predictions(sample)
        total_time = time.time() - start

        out_path = Path("data/processed/zero_shot_gpt4o_results.parquet")
        results_df.to_parquet(out_path, index=False)
        print(f"Saved raw results to {out_path}")

        n_errors = results_df["error"].notna().sum()
        total_prompt_tokens = results_df["prompt_tokens"].sum()
        total_completion_tokens = results_df["completion_tokens"].sum()
        avg_latency = results_df["latency_s"].dropna().mean()
        cost = (
            total_prompt_tokens / 1_000_000 * PRICE_PER_MTOK_INPUT
            + total_completion_tokens / 1_000_000 * PRICE_PER_MTOK_OUTPUT
        )
        cost_per_1000 = cost / len(results_df) * 1000

        print(f"\n--- Cost & latency ({len(results_df)} requests, {n_errors} errors) ---")
        print(f"Total wall time: {total_time:.0f}s ({MAX_WORKERS} concurrent workers)")
        print(f"Avg per-request latency: {avg_latency:.2f}s")
        print(f"Total tokens: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion")
        print(f"Estimated cost: ${cost:.4f} total, ${cost_per_1000:.4f} per 1000 requests")

        mlflow.log_metric("avg_latency_s", avg_latency)
        mlflow.log_metric("total_cost_usd", cost)
        mlflow.log_metric("cost_per_1000_requests_usd", cost_per_1000)
        mlflow.log_metric("n_errors", int(n_errors))

        evaluate_head(results_df, "product", "product", "product_pred")
        evaluate_head(results_df, "priority", "priority_bucket", "priority_pred")

    print("\nDone. View results with: uv run mlflow ui")


if __name__ == "__main__":
    main()
