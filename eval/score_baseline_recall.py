"""Score baseline contexts with ContextRecall only, then compare to enhanced.

Reads eval/ragas_inputs_baseline.jsonl and eval/ragas_scores.json (enhanced),
writes eval/ragas_recall_comparison.json with both scores and improvement.
Both evaluation rules (relative and absolute) are reported; no single threshold
interpretation is assumed.
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from eval import run_ragas as R  # noqa: E402
from ragas.metrics.collections import ContextRecall  # noqa: E402

BASELINE_INPUT = ROOT / "eval" / "ragas_inputs_baseline.jsonl"
ENHANCED_SCORES = ROOT / "eval" / "ragas_scores.json"
OUTPUT = ROOT / "eval" / "ragas_recall_comparison.json"


async def main():
    records = [json.loads(l) for l in BASELINE_INPUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    refs = json.loads(R.REFERENCES.read_text(encoding="utf-8"))
    llm = R.build_llm()
    enhanced = json.loads(ENHANCED_SCORES.read_text(encoding="utf-8"))
    out = {}
    for rec in records:
        qid = rec["id"]
        if not rec["contexts"]:
            continue
        metric = ContextRecall(llm=llm)
        try:
            result = await asyncio.wait_for(
                metric.ascore(user_input=rec["question"], retrieved_contexts=rec["contexts"],
                              reference=refs[qid]), timeout=600)
            baseline_score = result.value
        except Exception as exc:
            print(f"{qid} baseline recall ERROR {type(exc).__name__}", flush=True)
            baseline_score = None
        enhanced_score = enhanced.get(qid, {}).get("context_recall")
        row = {"baseline": baseline_score, "enhanced": enhanced_score}
        if baseline_score is not None and enhanced_score is not None:
            row["absolute_gain"] = round(enhanced_score - baseline_score, 4)
            row["relative_gain"] = (round((enhanced_score - baseline_score) / baseline_score, 4)
                                    if baseline_score > 0 else None)
        out[qid] = row
        print(f"{qid}: baseline={baseline_score} enhanced={enhanced_score} "
              f"abs={row.get('absolute_gain')} rel={row.get('relative_gain')}", flush=True)
    OUTPUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
