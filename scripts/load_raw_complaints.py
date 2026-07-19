import os
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = Path("data/raw/complaints.csv")
SQL_DIR = Path("sql")


def connect():
    return psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def run_sql_file(cur, path: Path):
    cur.execute(path.read_text())


def main():
    conn = connect()
    conn.autocommit = False

    with conn.cursor() as cur:
        print("Dropping and recreating table...")
        cur.execute("DROP TABLE IF EXISTS raw_complaints;")
        run_sql_file(cur, SQL_DIR / "001_raw_complaints_schema.sql")
        conn.commit()

        print(f"Copying {CSV_PATH} into raw_complaints ...")
        start = time.time()
        with open(CSV_PATH, "rb") as f, cur.copy(
            "COPY raw_complaints FROM STDIN WITH (FORMAT csv, HEADER true)"
        ) as copy:
            while chunk := f.read(1024 * 1024):
                copy.write(chunk)
        conn.commit()
        print(f"COPY finished in {time.time() - start:.0f}s")

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM raw_complaints;")
        (count,) = cur.fetchone()
        print(f"Row count in raw_complaints: {count:,}")

    with conn.cursor() as cur:
        print("Building indexes...")
        start = time.time()
        run_sql_file(cur, SQL_DIR / "002_raw_complaints_indexes.sql")
        conn.commit()
        print(f"Indexes built in {time.time() - start:.0f}s")

    conn.close()


if __name__ == "__main__":
    main()
