"""Repeat ContextRecall scoring for enhanced and baseline contexts (N runs each).

Because single scores are unstable with the evaluation LLM, this averages N runs
per question per variant, using the same question/reference pairs. Writes
eval/ragas_recall_averages.json with means and per-run values.
"""
import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from eval import run_ragas as R  # noqa: E402
from ragas.metrics.collections import ContextRecall  # noqa: E402

BASELINE_INPUT = ROOT / "eval" / "ragas_inputs_baseline.jsonl"
OUTPUT = ROOT / "eval" / "ragas_recall_averages.json"
RUNS = 3


def load_inputs(path):
    return {json.loads(l)["id"]: json.loads(l)
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


async def score_one(llm, record, reference):
    return (await asyncio.wait_for(
        ContextRecall(llm=llm).ascore(user_input=record["question"],
                                      retrieved_contexts=record["contexts"],
                                      reference=reference), timeout=600)).value


async def main():
    enhanced = load_inputs(R.INPUTS)
    baseline = load_inputs(BASELINE_INPUT)
    refs = json.loads(R.REFERENCES.read_text(encoding="utf-8"))
    llm = R.build_llm()
    qids = sorted(set(enhanced) & set(baseline))
    out = {}
    for qid in qids:
        e_scores, b_scores = [], []
        for i in range(RUNS):
            for label, recs, scores in (("E", enhanced, e_scores), ("B", baseline, b_scores)):
                try:
                    scores.append(await score_one(llm, recs[qid], refs[qid]))
                except Exception as exc:
                    print(f"{qid} {label} run {i+1} ERROR {type(exc).__name__}", flush=True)
        e_mean = round(statistics.mean(e_scores), 4) if e_scores else None
        b_mean = round(statistics.mean(b_scores), 4) if b_scores else None
        row = {"enhanced_runs": e_scores, "baseline_runs": b_scores,
               "enhanced_mean": e_mean, "baseline_mean": b_mean}
        if e_mean is not None and b_mean is not None:
            row["absolute_gain"] = round(e_mean - b_mean, 4)
            row["relative_gain"] = (round((e_mean - b_mean) / b_mean, 4) if b_mean > 0 else None)
        out[qid] = row
        print(f"{qid}: E={e_scores} -> {e_mean} | B={b_scores} -> {b_mean} | "
              f"abs={row.get('absolute_gain')} rel={row.get('relative_gain')}", flush=True)
    OUTPUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
