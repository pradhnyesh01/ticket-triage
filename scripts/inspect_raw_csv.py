import csv
import sys
import time

csv.field_size_limit(sys.maxsize)

path = "data/raw/complaints.csv"
start = time.time()

total_rows = 0
narrative_present = 0

with open(path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    narrative_idx = header.index("Consumer complaint narrative")
    for row in reader:
        total_rows += 1
        if row[narrative_idx].strip():
            narrative_present += 1
        if total_rows % 2_000_000 == 0:
            elapsed = time.time() - start
            print(f"...{total_rows:,} rows in {elapsed:.0f}s", flush=True)

elapsed = time.time() - start
print(f"\nheader ({len(header)} columns): {header}")
print(f"total data rows: {total_rows:,}")
print(f"rows with non-empty narrative: {narrative_present:,} ({100 * narrative_present / total_rows:.1f}%)")
print(f"elapsed: {elapsed:.0f}s")
