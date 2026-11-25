"""Comparison experiments on recorded Acemap snapshots.

Subcommands (each writes a separate JSON file under eval/):
  three-modes      same questions across novice/intermediate/expert
  personalization  same question with and without interests
  variability      repeated runs to quantify non-determinism
  baseline         enhanced retrieval vs pure-Acemap baseline (data only)

All runs use replay snapshots. LLM steps still run live with temperature 0;
results are recorded as-is, without caching or post-processing.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import rag_system  # noqa: E402
from AcemapPilot.acemap import fetch  # noqa: E402


def replay(term, size, *, order_by=None):
    return fetch(term, size=size, replay=True, order_by=order_by)


def run(question, mode, interests=(), baseline=False):
    result = rag_system.enhanced_retrieve(
        question, mode=mode, interests=list(interests), baseline=baseline, paper_source=replay)
    papers = result["papers"]
    return {
        "status": result["generation_status"],
        "papers": len(papers),
        "ids": [p["id"] for p in papers],
        "titles": [str(p.get("title", ""))[:40] for p in papers],
        "years": [p.get("year") for p in papers],
        "recommended": [p["id"] for p in papers if p.get("recommended")],
        "warnings": result["warnings"],
        "steps": {s["stage"]: s["detail"] for s in result["steps"]},
        "retrieval": result["retrieval"],
    }


def save(name, data):
    path = ROOT / "eval" / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {path}", flush=True)


def three_modes():
    questions = [
        "板块构造近五年有哪些研究进展？",
        "RAG 近五年有哪些研究进展",
        "气候变化近五年有哪些研究进展",
    ]
    out = {}
    for q in questions:
        out[q] = {}
        for mode in ("novice", "intermediate", "expert"):
            try:
                out[q][mode] = run(q, mode)
                r = out[q][mode]
                print(f"{mode:<12} {q[:18]:<20} papers={r['papers']} status={r['status']}", flush=True)
            except rag_system.SearchError as exc:
                out[q][mode] = {"error": str(exc)[:200]}
                print(f"{mode:<12} {q[:18]:<20} ERROR", flush=True)
    save("compare_three_modes.json", out)


def personalization():
    combos = [
        ("板块构造近五年有哪些研究进展？", "expert", (12,)),
        ("板块构造近五年有哪些研究进展？", "intermediate", (12,)),
        ("板块构造近五年有哪些研究进展？", "novice", (12,)),
    ]
    out = {}
    for question, mode, interests in combos:
        key = f"{mode}|{','.join(map(str, interests))}"
        out[key] = {}
        for label, use in (("with", interests), ("without", ())):
            try:
                out[key][label] = run(question, mode, use)
                r = out[key][label]
                print(f"{mode:<12} {label:<8} papers={r['papers']} "
                      f"recommended={len(r['recommended'])}", flush=True)
            except rag_system.SearchError as exc:
                out[key][label] = {"error": str(exc)[:200]}
    for key, pair in out.items():
        if "with" in pair and "without" in pair and "ids" in pair["with"] and "ids" in pair["without"]:
            a, b = pair["with"], pair["without"]
            pair["same_set"] = set(a["ids"]) == set(b["ids"])
            pair["same_order"] = a["ids"] == b["ids"]
            print(f"{key}: same_set={pair['same_set']} same_order={pair['same_order']}", flush=True)
    save("compare_personalization.json", out)


def variability():
    questions = ["N3:subduction 是什么", "E4:climate change 近五年的最新进展", "I1:板块构造近五年有哪些研究进展"]
    rounds = 5
    out = {}
    for item in questions:
        qid, question = item.split(":", 1)
        mode = qid[0].lower() if qid[0].isalpha() else "intermediate"
        out[qid] = {"question": question, "mode": mode, "runs": []}
        for i in range(rounds):
            try:
                r = run(question, mode)
                out[qid]["runs"].append({"papers": r["papers"], "status": r["status"],
                                         "ids": r["ids"][:5]})
                print(f"{qid} run {i+1}: {r['papers']} papers, {r['status']}", flush=True)
            except rag_system.SearchError as exc:
                out[qid]["runs"].append({"error": str(exc)[:120]})
    for qid, data in out.items():
        counts = [r.get("papers") for r in data["runs"] if "papers" in r]
        if counts:
            data["min"] = min(counts)
            data["max"] = max(counts)
            data["mean"] = round(statistics.mean(counts), 2)
            data["distinct_counts"] = sorted(set(counts))
            print(f"{qid}: counts={counts} distinct={data['distinct_counts']}", flush=True)
    save("compare_variability.json", out)


def baseline():
    questions = [
        "板块构造近五年有哪些研究进展？",
        "RAG 近五年有哪些研究进展",
        "气候变化近五年有哪些研究进展",
    ]
    out = {}
    for q in questions:
        out[q] = {"enhanced": run(q, "expert"), "baseline": run(q, "expert", baseline=True)}
        e, b = out[q]["enhanced"], out[q]["baseline"]
        out[q]["same_set"] = set(e["ids"]) == set(b["ids"])
        out[q]["only_enhanced"] = sorted(set(e["ids"]) - set(b["ids"]))
        out[q]["only_baseline"] = sorted(set(b["ids"]) - set(e["ids"]))
        print(f"{q[:18]:<20} enhanced={e['papers']} baseline={b['papers']} "
              f"same_set={out[q]['same_set']}", flush=True)
    save("compare_baseline.json", out)


if __name__ == "__main__":
    {"three-modes": three_modes, "personalization": personalization,
     "variability": variability, "baseline": baseline}[sys.argv[1]]()
