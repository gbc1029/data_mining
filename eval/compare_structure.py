"""Structural comparison without any LLM call.

Patches select_relevant_evidence to accept all candidates and _conclude to a
placeholder, so differences come only from retrieval, filtering and ranking.
Useful while the DashScope free quota is exhausted. Does NOT measure answer
quality or LLM-based relevance -- results are structural only.
"""
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import rag_system  # noqa: E402
from AcemapPilot.acemap import fetch  # noqa: E402


def replay(term, size, *, order_by=None):
    return fetch(term, size=size, replay=True, order_by=order_by)


def accept_all(question, topic, evidence, *, overview=False):
    # The real function converts 1-based JSON ids to 0-based indices.
    return set(range(len(evidence)))


def run_without_llm(question, mode, interests=(), baseline=False):
    with mock.patch.object(rag_system, "select_relevant_evidence", accept_all), \
         mock.patch.object(rag_system, "_conclude", lambda *a, **kw: "（占位，未调用 LLM）"):
        result = rag_system.enhanced_retrieve(
            question, mode=mode, interests=list(interests), baseline=baseline, paper_source=replay)
    papers = result["papers"]
    return {
        "status": result["generation_status"],
        "papers": len(papers),
        "ids": [p["id"] for p in papers],
        "recommended": [p["id"] for p in papers if p.get("recommended")],
        "years": [p.get("year") for p in papers],
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
            r = run_without_llm(q, mode)
            out[q][mode] = r
            print(f"{mode:<12} {q[:18]:<20} papers={r['papers']} "
                  f"terms={r['retrieval']['terms']}", flush=True)
        modes = list(out[q])
        for i, a in enumerate(modes):
            for b in modes[i + 1:]:
                sa, sb = set(out[q][a]["ids"]), set(out[q][b]["ids"])
                print(f"   {a} vs {b}: same_set={sa == sb} "
                      f"only_a={len(sa - sb)} only_b={len(sb - sa)}", flush=True)
    save("compare_structure_three_modes.json", out)


def personalization():
    combos = [
        ("板块构造近五年有哪些研究进展？", "expert", (12,)),
        ("板块构造近五年有哪些研究进展？", "intermediate", (12,)),
        ("板块构造近五年有哪些研究进展？", "novice", (12,)),
    ]
    out = {}
    for question, mode, interests in combos:
        key = mode
        out[key] = {"with": run_without_llm(question, mode, interests),
                    "without": run_without_llm(question, mode)}
        a, b = out[key]["with"], out[key]["without"]
        out[key]["same_set"] = set(a["ids"]) == set(b["ids"])
        out[key]["same_order"] = a["ids"] == b["ids"]
        out[key]["only_with"] = sorted(set(a["ids"]) - set(b["ids"]))
        out[key]["only_without"] = sorted(set(b["ids"]) - set(a["ids"]))
        print(f"{mode:<12} with={a['papers']} without={b['papers']} "
              f"same_set={out[key]['same_set']} same_order={out[key]['same_order']} "
              f"recommended_with={len(a['recommended'])}", flush=True)
    save("compare_structure_personalization.json", out)


def baseline():
    questions = [
        "板块构造近五年有哪些研究进展？",
        "RAG 近五年有哪些研究进展",
        "气候变化近五年有哪些研究进展",
    ]
    out = {}
    for q in questions:
        out[q] = {"enhanced": run_without_llm(q, "expert"),
                  "baseline": run_without_llm(q, "expert", baseline=True)}
        e, b = out[q]["enhanced"], out[q]["baseline"]
        out[q]["same_set"] = set(e["ids"]) == set(b["ids"])
        out[q]["only_enhanced"] = sorted(set(e["ids"]) - set(b["ids"]))
        out[q]["only_baseline"] = sorted(set(b["ids"]) - set(e["ids"]))
        out[q]["recall_gain_count"] = len(set(e["ids"]) - set(b["ids"]))
        print(f"{q[:18]:<20} enhanced={e['papers']} baseline={b['papers']} "
              f"only_enhanced={out[q]['recall_gain_count']}", flush=True)
    save("compare_structure_baseline.json", out)


def determinism():
    q = "板块构造近五年有哪些研究进展？"
    runs = [run_without_llm(q, "expert") for _ in range(3)]
    out = {"question": q, "runs": runs,
           "all_identical": all(r["ids"] == runs[0]["ids"] for r in runs)}
    print("structural determinism (expert):", out["all_identical"],
          "counts:", [r["papers"] for r in runs], flush=True)
    save("compare_structure_determinism.json", out)


if __name__ == "__main__":
    {"three-modes": three_modes, "personalization": personalization,
     "baseline": baseline, "determinism": determinism}[sys.argv[1]]()
