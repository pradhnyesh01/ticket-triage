-- Composite priority/urgency label (Low/Medium/High), engineered from
-- post-hoc OUTCOME fields only known after a complaint was handled.
-- This is the label; it must never be used as a model input (see the
-- leakage test in tests/). Rubric:
--   +2  company_response_to_consumer = 'Closed with monetary relief'
--   +2  timely_response = 'No'
--   +1  tags mentions 'Older American' or 'Servicemember'
--   +1  issue/sub_issue in the curated high-severity list below
-- Bucketed: Low (0-1), Medium (2-3), High (4+).
--
-- Column note: the build plan's rubric header says "Company public
-- response includes monetary relief", but the actual condition text
-- ("Closed with monetary relief") is a value of
-- company_response_to_consumer, not company_public_response (a
-- different column -- the company's optional public statement, whose
-- values look like "Company believes it acted appropriately...").
-- Verified against real data before implementing.
--
-- High-severity list note: matched against real (issue, sub_issue)
-- values in the loaded data, not assumed from category names alone.
-- Two categories were deliberately EXCLUDED despite sounding related:
--   - 'Problem with fraud alerts or security freezes' -- friction
--     managing a protective feature, not evidence of actual fraud.
--   - 'Threatened to contact someone or share information improperly'
--     -- improper third-party disclosure/harassment by a collector,
--     a different harm than a threat of legal/negative action.
-- Legacy issue-taxonomy names (CFPB has renamed categories over the
-- years) are matched alongside their current equivalents.
CREATE OR REPLACE VIEW priority_label AS
WITH scored AS (
    SELECT
        complaint_id,
        (CASE WHEN company_response_to_consumer = 'Closed with monetary relief' THEN 2 ELSE 0 END)
      + (CASE WHEN timely_response = 'No' THEN 2 ELSE 0 END)
      + (CASE WHEN tags ILIKE '%Older American%' OR tags ILIKE '%Servicemember%' THEN 1 ELSE 0 END)
      + (CASE WHEN
            issue IN (
                'Fraud or scam',
                'Identity theft / Fraud / Embezzlement',
                'Attempts to collect debt not owed',
                'Cont''d attempts collect debt not owed',
                'Took or threatened to take negative or legal action',
                'Taking/threatening an illegal action'
            )
            OR (issue, sub_issue) IN (
                ('Getting a credit card', 'Card opened as result of identity theft or fraud'),
                ('Opening an account', 'Account opened as a result of fraud'),
                ('Getting a loan or lease', 'Fraudulent loan'),
                ('Getting a loan', 'Fraudulent loan'),
                ('Getting a loan or lease', 'Credit denial'),
                ('Communication tactics', 'Threatened to take legal action')
            )
         THEN 1 ELSE 0 END)
        AS priority_score
    FROM narrative_consented
)
SELECT
    complaint_id,
    priority_score,
    CASE
        WHEN priority_score >= 4 THEN 'High'
        WHEN priority_score >= 2 THEN 'Medium'
        ELSE 'Low'
    END AS priority_bucket
FROM scored;
