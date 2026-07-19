-- Rows usable for modeling: a real, non-null complaint_id (see the raw
-- ingestion note in 001 -- ~0.02% of rows arrive without one) and a
-- non-empty consumer narrative. The CFPB export gates the narrative
-- column itself on consumer opt-in + publication approval, so "narrative
-- present" IS the consent filter; there is no separate consent column.
CREATE OR REPLACE VIEW narrative_consented AS
SELECT *
FROM raw_complaints
WHERE complaint_id IS NOT NULL
  AND consumer_complaint_narrative IS NOT NULL
  AND btrim(consumer_complaint_narrative) <> '';
