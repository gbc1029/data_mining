"""Record Acemap responses for the fixed evaluation questions.

Runs each question through the real enhanced_retrieve path with a recording
paper_source, so the recorded (term, size, order_by) triples are exactly the ones
the live pipeline requests. Writes eval/index.csv for replay.
"""
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load C/.env before the pipeline modules read their environment at import time.
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import rag_system  # noqa: E402
from AcemapPilot.acemap import fetch  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.csv"
INDEX = ROOT / "eval" / "index.csv"
GAP_SECONDS = 2.0


RETRIES = 4
BACKOFF_SECONDS = 5.0


def record_source(log):
    """Record one Acemap request, retrying transient connection resets and timeouts only."""
    def source(term, size, *, order_by=None):
        last = None
        for attempt in range(1, RETRIES + 1):
            try:
                rows = fetch(term, size=size, save=True, order_by=order_by)
                log.append({"term": term, "size": size, "order_by": order_by or "",
                            "n": len(rows), "attempts": attempt})
                time.sleep(GAP_SECONDS)
                return rows
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = exc
                print(f"    retry {attempt}/{RETRIES} for {term!r}: {type(exc).__name__}", flush=True)
                time.sleep(BACKOFF_SECONDS * attempt)
        raise last
    return source


def main():
    rows_out = []
    with QUESTIONS.open(encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            interests = [int(x) for x in item["interests"].split() if x] if item["interests"] else []
            log = []
            try:
                result = rag_system.enhanced_retrieve(
                    item["question"], mode=item["mode"], interests=interests,
                    paper_source=record_source(log),
                )
                status = result.get("generation_status", "")
                papers = len(result.get("papers", []))
                error = ""
            except rag_system.SearchError as exc:
                status, papers, error = "search_error", 0, str(exc)[:200]
            for entry in log:
                rows_out.append({"qid": item["id"], "mode": item["mode"], **entry})
            print(f"{item['id']:<3} {item['mode']:<12} requests={len(log)} papers={papers} "
                  f"status={status} {error}", flush=True)

    # One row per snapshot file: a term requested by several questions is recorded once.
    seen, unique = set(), []
    for row in rows_out:
        key = (row["term"], row["size"], row["order_by"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    with INDEX.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "mode", "term", "size", "order_by", "n", "attempts"])
        writer.writeheader()
        writer.writerows(unique)
    print(f"wrote {len(unique)} unique snapshots ({len(rows_out)} requests) to {INDEX}", flush=True)


if __name__ == "__main__":
    main()
