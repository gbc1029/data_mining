"""Score exported answers with RAGAS 0.4.4.

Evaluation LLM: qwen3.8-max-0902 via DashScope's OpenAI-compatible endpoint.
Embeddings: local sentence-transformers all-MiniLM-L6-v2 (no network).
Metrics: Faithfulness, AnswerRelevancy, ContextRecall (needs a reference).

Reads eval/ragas_inputs.jsonl (question, answer, contexts) and optional
eval/references.json ({"id": "reference answer"}). Writes eval/ragas_scores.json.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import openai  # noqa: E402
from ragas.embeddings import HuggingFaceEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness  # noqa: E402

EVAL_MODEL = os.getenv("RAGAS_EVAL_MODEL", "kimi-k3.1-pro")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
INPUTS = ROOT / "eval" / "ragas_inputs.jsonl"
REFERENCES = ROOT / "eval" / "references.json"
OUTPUT = ROOT / "eval" / "ragas_scores.json"
MINILM_DIR = ROOT / ".runtime" / "huggingface" / "hub"


def build_llm():
    key = os.getenv("RAGAS_EVAL_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        raise SystemExit("未找到评估服务密钥（DASHSCOPE_API_KEY）。")
    client = openai.AsyncOpenAI(base_url=DASHSCOPE_URL, api_key=key)
    # ragas defaults to max_tokens=1024; structured metric outputs are routinely
    # truncated at that limit (IncompleteOutputException / finish_reason=length).
    # kimi-k3 only accepts temperature=0 (other values, including 0.01 and
    # top_p overrides, are rejected). Explicit 0 gives deterministic scoring;
    # omitting the parameter lets the service apply a non-zero default and
    # causes score drift between identical runs.
    # extra_body is the OpenAI-SDK-compatible way to reach DashScope's
    # enable_thinking switch without adding an unsupported top-level kwarg.
    return llm_factory(EVAL_MODEL, provider="openai", client=client, max_tokens=4096,
                       temperature=0, top_p=None,
                       extra_body={"enable_thinking": False})


def build_embeddings():
    # Modern ragas interface backed by the local sentence-transformers MiniLM (no API).
    return HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",
                                 use_api=False, device="cpu")


async def score_one(record, reference, llm, embeddings):
    faith = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    recall = ContextRecall(llm=llm)
    contexts = record["contexts"]
    result = {}
    f = await faith.ascore(user_input=record["question"], response=record["answer"],
                           retrieved_contexts=contexts)
    result["faithfulness"] = f.value
    r = await relevancy.ascore(user_input=record["question"], response=record["answer"])
    result["answer_relevancy"] = r.value
    if reference:
        c = await recall.ascore(user_input=record["question"], retrieved_contexts=contexts,
                                reference=reference)
        result["context_recall"] = c.value
    else:
        result["context_recall"] = None
    return result


async def main():
    if not INPUTS.exists():
        raise SystemExit("缺少 eval/ragas_inputs.jsonl，请先运行 export_ragas_inputs.py。")
    records = [json.loads(line) for line in INPUTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    # E3 is recorded but excluded from evaluation by decision (no contexts; not a judgement basis).
    records = [r for r in records if r["id"] != "E3" and r["contexts"]]
    references = json.loads(REFERENCES.read_text(encoding="utf-8")) if REFERENCES.exists() else {}
    llm = build_llm()
    embeddings = build_embeddings()
    scores = {}
    for record in records:
        try:
            scores[record["id"]] = await score_one(record, references.get(record["id"]), llm, embeddings)
            print(record["id"], scores[record["id"]], flush=True)
        except Exception as exc:  # keep going; record the failure per question
            scores[record["id"]] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(record["id"], "ERROR", scores[record["id"]]["error"], flush=True)
    OUTPUT.write_text(json.dumps(scores, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
