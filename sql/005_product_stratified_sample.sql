-- ~250K subsample, proportionally stratified by Product: each product
-- keeps roughly the same share of the sample as it has in the full
-- narrative-consented population, so real-world class imbalance is
-- preserved (and handled deliberately later via class weighting/split
-- strategy) rather than hidden by artificial rebalancing at this stage.
--
-- Materialized, not a plain view: plain random() sampling would draw a
-- DIFFERENT 250K rows every time the view is queried, silently breaking
-- train/val/test split stability across later pipeline steps. Row order
-- is a deterministic pseudo-random hash (complaint_id salted with a
-- fixed string), not Postgres's random(), so the exact same sample
-- results every time this is rebuilt from the same inputs.
CREATE MATERIALIZED VIEW IF NOT EXISTS product_stratified_sample AS
WITH ranked AS (
    SELECT
        nc.*,
        pl.priority_score,
        pl.priority_bucket,
        ROW_NUMBER() OVER (
            PARTITION BY nc.product
            ORDER BY md5(nc.complaint_id::text || ':ticket-triage-sample-v1')
        ) AS rn,
        COUNT(*) OVER (PARTITION BY nc.product) AS product_total,
        COUNT(*) OVER () AS grand_total
    FROM narrative_consented nc
    JOIN priority_label pl USING (complaint_id)
)
SELECT
    complaint_id, date_received, product, sub_product, issue, sub_issue,
    consumer_complaint_narrative, company_public_response, company, state,
    zip_code, tags, submitted_via, date_sent_to_company,
    company_response_to_consumer, timely_response,
    priority_score, priority_bucket
FROM ranked
WHERE rn <= CEIL(product_total::numeric / grand_total * 250000);

CREATE UNIQUE INDEX IF NOT EXISTS idx_product_stratified_sample_complaint_id
    ON product_stratified_sample (complaint_id);
CREATE INDEX IF NOT EXISTS idx_product_stratified_sample_product
    ON product_stratified_sample (product);
