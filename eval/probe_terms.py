"""Probe which Acemap terms a given question/mode/interest combination requests.

Uses a fake paper source that records terms and returns nothing, so no network or
LLM-visible data is used beyond topic resolution. Reports which recorded snapshots
are missing, so comparison runs can be recorded in advance.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import rag_system  # noqa: E402
from AcemapPilot.acemap import snapshot_path  # noqa: E402


def probe(question, mode, interests=(), baseline=False):
    seen = []

    def fake_source(term, size, *, order_by=None):
        seen.append({"term": term, "size": size, "order_by": order_by or ""})
        return []

    try:
        rag_system.enhanced_retrieve(question, mode=mode, interests=list(interests),
                                     baseline=baseline, paper_source=fake_source)
    except rag_system.SearchError:
        pass  # expected: empty source means all terms "failed"
    missing = []
    for call in seen:
        path = snapshot_path(call["term"], call["size"], order_by=call["order_by"] or None)
        if not path.exists():
            missing.append(call)
    return seen, missing


def main():
    combos = [
        ("板块构造近五年有哪些研究进展？", "novice", (), False),
        ("板块构造近五年有哪些研究进展？", "intermediate", (), False),
        ("板块构造近五年有哪些研究进展？", "expert", (), False),
        ("板块构造近五年有哪些研究进展？", "expert", (12,), False),
        ("RAG 近五年有哪些研究进展", "novice", (), False),
        ("RAG 近五年有哪些研究进展", "expert", (), False),
        ("气候变化近五年有哪些研究进展", "novice", (), False),
        ("气候变化近五年有哪些研究进展", "expert", (), False),
        ("板块构造近五年有哪些研究进展？", "expert", (), True),
        ("RAG 近五年有哪些研究进展", "expert", (), True),
        ("气候变化近五年有哪些研究进展", "expert", (), True),
    ]
    out = []
    for question, mode, interests, baseline in combos:
        seen, missing = probe(question, mode, interests, baseline)
        row = {"question": question, "mode": mode, "interests": list(interests),
               "baseline": baseline, "terms": seen, "missing": missing}
        out.append(row)
        print(f"{mode:<12} baseline={baseline!s:<5} interests={list(interests)} | "
              f"{question[:20]:<22} terms={len(seen)} missing={len(missing)}", flush=True)
        for call in missing:
            print(f"    MISSING {call['term']!r} order_by={call['order_by']!r}", flush=True)
    (ROOT / "eval" / "probe_terms.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
