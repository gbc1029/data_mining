"""Retry only the RAGAS questions that errored, saving after each success.

Each question gets a bounded time budget so one stalled call cannot block the rest.
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

PER_QUESTION_TIMEOUT = 300  # seconds
ATTEMPTS = 2


def load_scores():
    return json.loads(R.OUTPUT.read_text(encoding="utf-8"))


def save_scores(scores):
    R.OUTPUT.write_text(json.dumps(scores, ensure_ascii=False, indent=1), encoding="utf-8")


async def main():
    recs = {json.loads(l)["id"]: json.loads(l)
            for l in R.INPUTS.read_text(encoding="utf-8").splitlines() if l.strip()}
    refs = json.loads(R.REFERENCES.read_text(encoding="utf-8")) if R.REFERENCES.exists() else {}
    scores = load_scores()
    pending = [k for k, v in scores.items() if "error" in v]
    print("retrying:", pending, flush=True)
    llm = R.build_llm()
    emb = R.build_embeddings()
    for qid in pending:
        for attempt in range(1, ATTEMPTS + 1):
            print(f"{qid} attempt {attempt} start", flush=True)
            try:
                result = await asyncio.wait_for(
                    R.score_one(recs[qid], refs.get(qid), llm, emb), timeout=PER_QUESTION_TIMEOUT)
                scores[qid] = result
                save_scores(scores)
                print(f"{qid} ok {result}", flush=True)
                break
            except asyncio.TimeoutError:
                print(f"{qid} attempt {attempt} timed out after {PER_QUESTION_TIMEOUT}s", flush=True)
            except Exception as exc:
                print(f"{qid} attempt {attempt} {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                await asyncio.sleep(5)
        else:
            scores[qid] = {"error": "failed after retries; see retry_failed_ragas.py"}
            save_scores(scores)
    still = [k for k, v in scores.items() if "error" in v]
    print("done. still failing:", still, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
