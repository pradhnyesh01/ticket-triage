from ticket_triage.features import LEAKAGE_COLUMNS, extract_model_features

# Shaped like a row from product_stratified_sample: every column a model
# could theoretically see, including the outcome fields the priority
# label is built from.
SAMPLE_ROW = {
    "complaint_id": 123,
    "date_received": "2026-01-01",
    "product": "Debt collection",
    "sub_product": None,
    "issue": "Attempts to collect debt not owed",
    "sub_issue": "Debt is not mine",
    "consumer_complaint_narrative": "They keep calling me about a debt that isn't mine.",
    "company_public_response": "Company disputes the facts presented in the complaint",
    "company": "ACME Collections",
    "state": "CA",
    "zip_code": "90210",
    "tags": "Older American",
    "submitted_via": "Web",
    "date_sent_to_company": "2026-01-02",
    "company_response_to_consumer": "Closed with monetary relief",
    "timely_response": "No",
    "priority_score": 4,
    "priority_bucket": "High",
}


def test_features_contain_no_leakage_columns():
    features = extract_model_features(SAMPLE_ROW)
    leaked = LEAKAGE_COLUMNS & features.keys()
    assert not leaked, f"Feature matrix leaked outcome columns: {leaked}"


def test_features_are_narrative_text_only():
    """Stronger than 'no leakage columns': per the build plan, narrative
    text is the ONLY input either head may ever receive -- not product,
    not dates, not company, nothing else."""
    features = extract_model_features(SAMPLE_ROW)
    assert set(features.keys()) == {"narrative"}
    assert features["narrative"] == SAMPLE_ROW["consumer_complaint_narrative"]
