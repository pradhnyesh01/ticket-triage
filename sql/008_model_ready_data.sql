-- Final, model-ready view: one row per sampled complaint, with the
-- canonical Product label, the Priority label, the train/val/test
-- assignment, and the narrative text -- and nothing else. Baseline (and
-- later fine-tuning) scripts should read from this view specifically,
-- so the leakage boundary (see tests/test_leakage.py) is enforced by
-- which columns even exist in the query result, not solely by
-- application code remembering to drop the outcome fields.
--
-- consumer_complaint_narrative is intentionally NOT renamed here: it
-- stays under its raw column name so extract_model_features() (see
-- src/ticket_triage/features.py) works identically against this view,
-- product_stratified_sample, or raw_complaints -- one contract, same
-- column name, everywhere a complaint row can come from. The "narrative"
-- key is what extract_model_features() produces as OUTPUT, not
-- something the view should pre-empt by renaming its input.
CREATE OR REPLACE VIEW model_ready_data AS
SELECT
    pss.complaint_id,
    pss.consumer_complaint_narrative,
    COALESCE(pcm.canonical_product, pss.product) AS product,
    pss.priority_bucket,
    ds.split
FROM product_stratified_sample pss
JOIN data_split ds USING (complaint_id)
LEFT JOIN product_canonical_map pcm ON pcm.raw_product = pss.product;
