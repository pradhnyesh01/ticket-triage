-- Raw CFPB Consumer Complaint Database export, loaded as-is via COPY.
-- Column order matches the CSV header exactly, since COPY relies on
-- positional order when no column list is given.
CREATE TABLE IF NOT EXISTS raw_complaints (
    date_received                 TIMESTAMPTZ,
    product                       TEXT,
    sub_product                   TEXT,
    issue                         TEXT,
    sub_issue                     TEXT,
    consumer_complaint_narrative  TEXT,
    company_public_response       TEXT,
    company                       TEXT,
    state                         TEXT,
    zip_code                      TEXT,
    tags                          TEXT,
    submitted_via                 TEXT,
    date_sent_to_company          TIMESTAMPTZ,
    company_response_to_consumer  TEXT,
    timely_response               TEXT,
    complaint_id                  BIGINT
);
