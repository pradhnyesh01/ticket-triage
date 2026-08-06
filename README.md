# Support Ticket Triage & Priority Classifier

A model that reads a free-text CFPB consumer complaint and predicts two things from the narrative alone: which **Product** category it belongs to, and how **urgent** it likely is. Built on the real, public [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) (17M+ rows, 8.5GB), not a synthetic or pre-labeled dataset.

The point of this project isn't the classifier itself — it's demonstrating a rigorous, documented evaluation harness (train and evaluate real models, not just call an LLM API) and defensible engineering decisions: label design under a real constraint, leakage avoidance, SQL-backed data handling at a scale where naive pandas actually breaks down.

## Problem statement

Given only the consumer's written complaint narrative — nothing else, not company name, not dates, not how the complaint was eventually resolved — predict:

- **Head A — Product**: which of 13 canonical product categories the complaint concerns (e.g. Debt collection, Mortgage, Credit reporting)
- **Head B — Priority**: Low / Medium / High urgency

Both heads are evaluated independently across four approaches: two classical TF-IDF baselines (Logistic Regression, XGBoost), a fine-tuned DistilBERT, and a zero-shot GPT-4o comparison — the same held-out test set, the same metrics, throughout.

## Why CFPB, and why the label had to be engineered

CFPB provides no urgency/priority field. Three Kaggle alternatives do ship a priority column, but none survive "where did this label come from?" — two have no published labeling methodology at all, and the one with documented provenance is fully synthetic (LLM-generated) and below the target scale. A label with no defensible origin is worse than no label. CFPB's real, government-collected data at real scale, with no ready-made answer, forces exactly the kind of judgment call this project exists to demonstrate.

### The priority rubric

Priority is an engineered composite score, computed in SQL (`sql/004_priority_label_view.sql`) from post-hoc **outcome** fields — signals only knowable after a complaint was handled, never available at intake:

| Signal | Condition | Points |
|---|---|---|
| Company gave monetary relief | `company_response_to_consumer = 'Closed with monetary relief'` | +2 |
| Untimely response | `timely_response = 'No'` | +2 |
| Vulnerable consumer | Tagged Older American or Servicemember | +1 |
| High-severity issue | Issue/Sub-issue in a curated list (fraud, identity theft, threats, debt not owed, credit denial) | +1 |

Bucketed: Low (0–1), Medium (2–3), High (4+). Two of the four signals are outcome-based (what happened *after* intake), two are contextual risk factors (known from the start). This is a **proxy label** — it reflects how the system responded to a complaint, not a human triager's judgment at intake — see [Limitations](#limitations).

The high-severity issue list was matched against real `(issue, sub_issue)` values in the data, not assumed from category names — two categories that sound related were deliberately excluded after review: `Problem with fraud alerts or security freezes` (friction with a protective feature, not evidence of actual fraud) and `Threatened to contact someone or share information improperly` (improper disclosure/harassment, a different harm than a legal threat).

### Leakage avoidance

The label is built from the four fields above; the model is never allowed to see them. `src/ticket_triage/features.py` defines the single function permitted to turn a database row into model input — it returns `{"narrative": ...}` and nothing else — and `tests/test_leakage.py` asserts this at two strengths: no named outcome column ever appears in the output, and the output's key set is *exactly* `{narrative}`, nothing else at all.

## Pipeline

PostgreSQL (Docker Compose) → SQL views (consent filtering → priority label → Product taxonomy canonicalization → stratified sample → Product×Priority-stratified train/val/test split) → baselines → DistilBERT fine-tuning (Colab) → zero-shot comparison → this evaluation. Full reasoning for every step, including three real bugs found and fixed along the way, is in the project's build log.

Scale: 17,010,343 raw complaints → 3,821,652 with a consented narrative → 250,010-row Product-stratified working sample → 175,010 train / 37,501 val / 37,499 test, stratified jointly on Product and Priority.

## Results

### Product (13 classes)

| Model | macro F1 | weighted F1 | notes |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.575 | 0.841 | 37s to train |
| TF-IDF + XGBoost | 0.571 | 0.842 | 28.4 min to train — no gain over LogReg for 46x the cost |
| **DistilBERT (fine-tuned)** | **0.614** | **0.860** | best result; pretrained semantic understanding helps most on the smallest categories |
| GPT-4o (zero-shot) | 0.515 | 0.790 | evaluated on a 1,272-row sample of the test set (see [Zero-shot sampling](#zero-shot-sampling)) |

### Priority (3 classes: Low / Medium / High)

| Model | macro F1 | High recall | High precision |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.405 | 0/34 (0%) | — |
| TF-IDF + XGBoost | 0.394 | 0/34 (0%) | — |
| DistilBERT (fine-tuned) | 0.439 | 0/34 (0%) | — |
| GPT-4o (zero-shot) | 0.051 | ~25/27 (93%) | 2% |

High priority is 0.09% of the data (3,551 of 3,821,652 narrative-consented complaints) — every approach struggles with it, in two opposite, mechanistically distinct ways (see [Error analysis](#qualitative-error-analysis)).

### Priority, binary (Low vs. Elevated = Medium-or-High)

The build plan's own named fallback if 3-way proved unworkable — tested empirically, not just assumed:

| Model | macro F1 | Elevated recall | Elevated precision |
|---|---|---|---|
| TF-IDF + Logistic Regression | **0.605** | **75%** | 19% |

A large, real improvement over the 3-way result (0.405 → 0.605), and — critically — an actually usable recall number instead of zero. A triage system using this would review roughly 5x more complaints than are truly elevated, but would catch 3 of every 4 genuinely elevated ones. That's a legitimate triage signal; 0% recall is not.

### Zero-shot cost & latency (GPT-4o, measured, not estimated)

1,532 requests attempted, 1,272 succeeded (260 lost to OpenAI rate-limiting under concurrent load — a script limitation, not a model result). Total cost **$1.69** (**$1.11 per 1,000 requests**), average latency 2.41s/request, total wall time 1,171s with 8 concurrent workers.

## Qualitative error analysis

26 misclassifications read directly (narrative text, true label, predicted label) across GPT-4o and Logistic Regression, focused on Priority's High class and Product's weakest categories.

**GPT-4o's false positives on Low (1,037 of 1,532 rows — the dominant error)** almost all contain the rubric's own high-severity vocabulary verbatim: "identity theft and fraud," "this is false and fraudulent," "I do not owe this amount," "credit denial." GPT-4o is correctly detecting the narrative-visible *issue-category* signal (worth +1 of the 4 points needed for High) and overweighting it into a full "High" prediction — reasonable as a reading of the text, but wrong for the label, because reaching High also requires the two outcome-based signals (monetary relief, untimely response) that genuinely cannot be inferred from what the consumer wrote.

**Logistic Regression's missed High cases (34 of 34 — every one)** show the same signal, opposite response: every single missed narrative contains explicit fraud/dispute language ("fraudulent demand," "false claim," "fraudulent charges"), and the model predicted Medium for 27 of them, Low for 7 — under by one tier, not off at random. The model has learned, correctly, that narratives like this typically *don't* clear the full 4-point threshold, because the other required points are close to independent of the text.

**GPT-4o's 7 missed true-High cases** are the most direct evidence for a real information ceiling: several have no severity language at all — a mundane stop-payment fee dispute, a landlord/cosigner paperwork issue, a deployment-related payment plan. These almost certainly reached High through the outcome-based signals alone (monetary relief, untimely response, or a vulnerable-consumer tag), with nothing in the text itself to signal it. No amount of reading comprehension recovers information that isn't present in the input.

**Product's `Credit card or prepaid card` misclassifications** (4 of 5 examined) were confused specifically with `Credit card` — the exact category pair flagged and deliberately left unmerged during taxonomy canonicalization (`sql/007_product_canonical_map.sql`), because their date ranges overlap rather than sequence cleanly. This reads as genuine ambiguity in the source taxonomy, not a model failure.

## Limitations

- **Priority is a proxy label, not ground truth.** It reflects how a complaint's resolution played out (company response, timeliness), not a human triager's judgment at intake. A production system would want real human-in-the-loop labels or historical triage logs instead. This project deliberately chose CFPB specifically *because* it forces this limitation into the open rather than hiding behind a pre-made label of unknown provenance.
- **High priority appears to have a real information ceiling from narrative text alone.** Three architecturally unrelated approaches (weighted classical ML, a fine-tuned transformer, a frontier zero-shot LLM) converge on failure, via two different mechanisms (conservative silence vs. indiscriminate over-calling) — not one model being under-powered. The qualitative analysis above traces this to two of the four rubric signals being determined by post-intake company behavior, not complaint content. The binary Low-vs-Elevated split recovers a genuinely usable signal (75% recall) where the 3-way split cannot.
- **Zero-shot results are on a smaller, non-representative sample** (1,272 of 37,499 test rows), with High deliberately over-represented (32 of 1,272 vs. 32 of 37,499 in the real test set) so it could be examined in detail. macro F1 and per-class precision/recall/F1 are valid for comparison regardless of this; weighted F1 and accuracy on this sample are not directly comparable to the other rows in the results tables.
- **Product taxonomy required manual reconciliation.** CFPB renamed its category taxonomy at least twice (2017, 2023); 4 families of "different" categories were confirmed via date-range analysis to be the same product and merged. One ambiguous case (Credit card / Prepaid card / Credit card or prepaid card) was deliberately left unmerged on weaker evidence, and shows up directly in the error analysis above.
- **17% of zero-shot requests failed** to OpenAI rate-limiting under this project's concurrency settings — a script limitation (fixable with a proper rate limiter), not a finding about the model.
