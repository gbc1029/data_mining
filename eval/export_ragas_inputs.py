"""Replay selected evaluation questions and export RAGAS inputs.

Writes eval/ragas_inputs.jsonl with one record per question:
question, answer (the generated conclusion), contexts (the retrieved passages),
mode, generation_status. Replay uses recorded Acemap snapshots only; the LLM
steps run live with temperature 0.
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
OUTPUT = ROOT / "eval" / "ragas_inputs.jsonl"


def replay(term, size, *, order_by=None):
    return fetch(term, size=size, replay=True, order_by=order_by)


def main(selected_ids):
    rows = list(csv.DictReader(QUESTIONS.open(encoding="utf-8-sig", newline="")))
    chosen = [row for row in rows if row["id"] in selected_ids]
    records = []
    for item in chosen:
        interests = [int(x) for x in item["interests"].split() if x] if item["interests"] else []
        try:
            result = rag_system.enhanced_retrieve(
                item["question"], mode=item["mode"], interests=interests, paper_source=replay
            )
        except rag_system.SearchError as exc:
            print(f"{item['id']} SearchError: {exc}", flush=True)
            continue
        records.append({
            "id": item["id"],
            "mode": item["mode"],
            "question": item["question"],
            "answer": result["conclusion"],
            "contexts": result["contexts"],
            "generation_status": result["generation_status"],
            "papers": len(result["papers"]),
        })
        print(f"{item['id']} status={result['generation_status']} papers={len(result['papers'])} "
              f"contexts={len(result['contexts'])}", flush=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"N2", "I1", "E1"})
