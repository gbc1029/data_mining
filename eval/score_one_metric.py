"""Score one (question, metric) pair and save it immediately.

Usage: python eval/score_one_metric.py <QID> <faithfulness|answer_relevancy|context_recall>
Each run is independent, so a slow or failed call never discards finished metrics.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from eval import run_ragas as R  # noqa: E402
from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness  # noqa: E402

TIMEOUT = 900


async def main(qid, metric):
    recs = {json.loads(l)["id"]: json.loads(l)
            for l in R.INPUTS.read_text(encoding="utf-8").splitlines() if l.strip()}
    rec = recs[qid]
    refs = json.loads(R.REFERENCES.read_text(encoding="utf-8"))
    llm = R.build_llm()
    t = time.time()
    if metric == "faithfulness":
        coro = Faithfulness(llm=llm).ascore(user_input=rec["question"], response=rec["answer"],
                                            retrieved_contexts=rec["contexts"])
    elif metric == "answer_relevancy":
        coro = AnswerRelevancy(llm=llm, embeddings=R.build_embeddings()).ascore(
            user_input=rec["question"], response=rec["answer"])
    else:
        coro = ContextRecall(llm=llm).ascore(user_input=rec["question"],
                                             retrieved_contexts=rec["contexts"], reference=refs[qid])
    value = (await asyncio.wait_for(coro, timeout=TIMEOUT)).value
    scores = json.loads(R.OUTPUT.read_text(encoding="utf-8"))
    scores.setdefault(qid, {})
    if "error" in scores[qid]:
        scores[qid] = {}
    scores[qid][metric] = value
    R.OUTPUT.write_text(json.dumps(scores, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{qid} {metric} = {value:.3f} ({round(time.time() - t)} s) saved", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
