"""Local cross-encoder ranking over selected evidence passages."""

from functools import cache
import re


CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


@cache
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(CROSS_MODEL, revision=CROSS_REVISION, device="cpu", max_length=384,
                        local_files_only=True, model_kwargs={"use_safetensors": True})


def cross_rank(question, topic, texts):
    if not texts:
        return []
    if re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", question) and topic.casefold() != question.casefold():
        question = f"{question} ({topic})"
    pairs = [(question, text) for text in texts.values()]
    scores = _cross_encoder().predict(pairs, batch_size=16, show_progress_bar=False)
    return sorted(texts, key=dict(zip(texts, scores, strict=True)).__getitem__, reverse=True)


def rerank_cached(question, topic, titles, passages):
    """Improve the shortlist if the English reranker is already cached."""
    if re.search(r"[\u3400-\u9fff]", question):
        return passages
    texts = {pid: f"{titles[pid]}\n{passages[pid]}" for pid in list(passages)[:40]}
    try:
        order = cross_rank(question, topic, texts)
    except (OSError, RuntimeError, ValueError):
        return passages
    return {**{pid: passages[pid] for pid in order}, **{pid: text for pid, text in passages.items() if pid not in texts}}
