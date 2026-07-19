-- Built after the bulk load (see scripts/load_raw_complaints.py). Creating
-- these before COPY would force Postgres to maintain them row-by-row across
-- 17M inserts; building them once after the data lands is much faster.
CREATE INDEX IF NOT EXISTS idx_raw_complaints_product ON raw_complaints (product);
CREATE INDEX IF NOT EXISTS idx_raw_complaints_date_received ON raw_complaints (date_received);
