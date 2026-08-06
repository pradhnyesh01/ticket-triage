"""SQL label logic tests.

Builds the priority-scoring rubric (sql/004_priority_label_view.sql)
against a small set of synthetic rows with known, controlled outcome
field values, so each point-scoring condition can be verified in
isolation -- the same rubric already spot-checked by hand against real
rows during development (see Build_Explanations.docx, Step 2), now
automated and repeatable.

Uses its own uniquely-named test table (test_priority_label_rows), not
raw_complaints/narrative_consented/priority_label -- this never reads
or writes the real pipeline's data, and is safe to run against the same
Postgres instance used for local development or a fresh, empty CI
Postgres service container.
"""

import os

import psycopg
import pytest
from dotenv import load_dotenv

load_dotenv()

TABLE = "test_priority_label_rows"

# The rubric, reimplemented against the test table's columns. Mirrors
# sql/004_priority_label_view.sql's scoring logic exactly -- if that
# file's conditions ever drift from this, these tests should be updated
# to match (and a mismatch here is a prompt to check which one is
# actually right, not to blindly edit whichever fails).
SCORE_SQL = f"""
    SELECT
        id,
        (CASE WHEN company_response_to_consumer = 'Closed with monetary relief' THEN 2 ELSE 0 END)
      + (CASE WHEN timely_response = 'No' THEN 2 ELSE 0 END)
      + (CASE WHEN tags ILIKE '%%Older American%%' OR tags ILIKE '%%Servicemember%%' THEN 1 ELSE 0 END)
      + (CASE WHEN issue IN ('Fraud or scam', 'Attempts to collect debt not owed') THEN 1 ELSE 0 END)
        AS priority_score
    FROM {TABLE}
    WHERE id = %(id)s;
"""


@pytest.fixture
def conn():
    connection = psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    with connection.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id INT PRIMARY KEY,
                company_response_to_consumer TEXT,
                timely_response TEXT,
                tags TEXT,
                issue TEXT
            );
            TRUNCATE {TABLE};
            """
        )
    connection.commit()
    yield connection
    with connection.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE};")
    connection.commit()
    connection.close()


def insert_row(conn, **kwargs):
    row = {
        "id": 1,
        "company_response_to_consumer": None,
        "timely_response": None,
        "tags": None,
        "issue": None,
        **kwargs,
    }
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {TABLE} (id, company_response_to_consumer, timely_response, tags, issue) "
            "VALUES (%(id)s, %(company_response_to_consumer)s, %(timely_response)s, %(tags)s, %(issue)s)",
            row,
        )
    conn.commit()


def score_of(conn, row_id=1) -> int:
    with conn.cursor() as cur:
        cur.execute(SCORE_SQL, {"id": row_id})
        return cur.fetchone()[1]


def test_no_signals_scores_zero(conn):
    insert_row(conn)
    assert score_of(conn) == 0


def test_monetary_relief_scores_two(conn):
    insert_row(conn, company_response_to_consumer="Closed with monetary relief")
    assert score_of(conn) == 2


def test_untimely_response_scores_two(conn):
    insert_row(conn, timely_response="No")
    assert score_of(conn) == 2


def test_timely_response_scores_zero(conn):
    insert_row(conn, timely_response="Yes")
    assert score_of(conn) == 0


def test_older_american_tag_scores_one(conn):
    insert_row(conn, tags="Older American")
    assert score_of(conn) == 1


def test_servicemember_tag_scores_one(conn):
    insert_row(conn, tags="Servicemember")
    assert score_of(conn) == 1


def test_combined_tags_still_score_one_not_two(conn):
    """Tags is a single +1 signal, not one point per tag present."""
    insert_row(conn, tags="Older American, Servicemember")
    assert score_of(conn) == 1


def test_high_severity_issue_scores_one(conn):
    insert_row(conn, issue="Fraud or scam")
    assert score_of(conn) == 1


def test_non_high_severity_issue_scores_zero(conn):
    insert_row(conn, issue="Communication tactics")
    assert score_of(conn) == 0


def test_all_four_signals_reach_high_threshold(conn):
    """The exact combination hand-verified against a real row (complaint_id
    1295840) during development (Build_Explanations.docx, Step 2):
    monetary relief (+2) + a TIMELY response, so no untimely bonus (+0)
    + Older American tag (+1) + high-severity issue (+1) = 4, crossing
    the High bucket's 4-point threshold."""
    insert_row(
        conn,
        company_response_to_consumer="Closed with monetary relief",
        timely_response="Yes",
        tags="Older American",
        issue="Attempts to collect debt not owed",
    )
    assert score_of(conn) == 4


def test_all_four_signal_types_present_scores_six(conn):
    """Distinct from the above: firing BOTH 2-point signals at once
    (monetary relief AND untimely response) alongside both 1-point
    signals sums to 6, not 4 -- this is its own combination, not the
    real row above, and confirms the four conditions add rather than
    cap or overwrite each other."""
    insert_row(
        conn,
        company_response_to_consumer="Closed with monetary relief",
        timely_response="No",
        tags="Servicemember",
        issue="Attempts to collect debt not owed",
    )
    assert score_of(conn) == 6
