import os
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()


def connect():
    return psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def main(paths):
    conn = connect()
    conn.autocommit = False
    with conn.cursor() as cur:
        for path in paths:
            print(f"Applying {path} ...")
            start = time.time()
            cur.execute(Path(path).read_text())
            conn.commit()
            print(f"  done in {time.time() - start:.1f}s")
    conn.close()


if __name__ == "__main__":
    main(sys.argv[1:])
