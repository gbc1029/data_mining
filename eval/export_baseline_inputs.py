"""Export baseline (pure Acemap) contexts for the evaluation questions.

Same questions and mode as the enhanced run, but baseline=True: one search term,
no graph expansion, no LLM relevance screening, Acemap order preserved.
Writes eval/ragas_inputs_baseline.jsonl (same schema as the enhanced export).
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import rag_system  # noqa: E402
from AcemapPilot.acemap import fetch  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.csv"
OUTPUT = ROOT / "eval" / "ragas_inputs_baseline.jsonl"


def replay(term, size, *, order_by=None):
    return fetch(term, size=size, replay=True, order_by=order_by)


def main():
    rows = [r for r in csv.DictReader(QUESTIONS.open(encoding="utf-8-sig", newline=""))
            if r["id"] != "E3"]
    records = []
    for item in rows:
        interests = [int(x) for x in item["interests"].split() if x] if item["interests"] else []
        try:
            result = rag_system.enhanced_retrieve(
                item["question"], mode=item["mode"], interests=interests,
                baseline=True, paper_source=replay)
        except rag_system.SearchError as exc:
            print(f"{item['id']} SearchError: {exc}", flush=True)
            continue
        records.append({
            "id": item["id"], "mode": item["mode"], "question": item["question"],
            "answer": result["conclusion"], "contexts": result["contexts"],
            "generation_status": result["generation_status"],
            "papers": len(result["papers"]), "baseline": True,
        })
        print(f"{item['id']} baseline papers={len(result['papers'])} "
              f"contexts={len(result['contexts'])} status={result['generation_status']}", flush=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
