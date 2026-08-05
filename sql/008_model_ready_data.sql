-- Final, model-ready view: one row per sampled complaint, with the
-- canonical Product label, the Priority label, the train/val/test
-- assignment, and the narrative text -- and nothing else. Baseline (and
-- later fine-tuning) scripts should read from this view specifically,
-- so the leakage boundary (see tests/test_leakage.py) is enforced by
-- which columns even exist in the query result, not solely by
-- application code remembering to drop the outcome fields.
CREATE OR REPLACE VIEW model_ready_data AS
SELECT
    pss.complaint_id,
    pss.consumer_complaint_narrative AS narrative,
    COALESCE(pcm.canonical_product, pss.product) AS product,
    pss.priority_bucket,
    ds.split
FROM product_stratified_sample pss
JOIN data_split ds USING (complaint_id)
LEFT JOIN product_canonical_map pcm ON pcm.raw_product = pss.product;
