-- Canonicalizes legacy CFPB Product taxonomy renames into their current
-- names. Verified via date-range analysis before writing this (see
-- Build_Explanations.docx): several "different" product categories are
-- actually the SAME product under sequential historical names, with
-- clean non-overlapping date cutovers, e.g.:
--   'Credit reporting'                                            2015-2017
--   '...credit repair services...'                                2017-2023
--   'Credit reporting or other personal consumer reports'         2023-2026
-- Without this, the model would be asked to distinguish between
-- categories that are really the same thing -- the credit-reporting
-- family alone is 65.7% of the dataset, fragmented across 3 "different"
-- labels by CFPB's own taxonomy history.
--
-- 'Credit card' / 'Prepaid card' / 'Credit card or prepaid card' are
-- deliberately NOT merged: their date ranges overlap rather than
-- sequence cleanly, so there isn't clean evidence they're the same
-- category -- left as three separate labels rather than guessed at.
CREATE OR REPLACE VIEW product_canonical_map AS
SELECT * FROM (VALUES
    ('Credit reporting', 'Credit reporting or other personal consumer reports'),
    ('Credit reporting, credit repair services, or other personal consumer reports', 'Credit reporting or other personal consumer reports'),
    ('Credit reporting or other personal consumer reports', 'Credit reporting or other personal consumer reports'),
    ('Bank account or service', 'Checking or savings account'),
    ('Checking or savings account', 'Checking or savings account'),
    ('Consumer Loan', 'Payday loan, title loan, personal loan, or advance loan'),
    ('Payday loan', 'Payday loan, title loan, personal loan, or advance loan'),
    ('Payday loan, title loan, or personal loan', 'Payday loan, title loan, personal loan, or advance loan'),
    ('Payday loan, title loan, personal loan, or advance loan', 'Payday loan, title loan, personal loan, or advance loan'),
    ('Money transfers', 'Money transfer, virtual currency, or money service'),
    ('Virtual currency', 'Money transfer, virtual currency, or money service'),
    ('Money transfer, virtual currency, or money service', 'Money transfer, virtual currency, or money service')
) AS t(raw_product, canonical_product);
