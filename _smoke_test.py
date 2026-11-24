"""Offline smoke test for GraphRAG4Scholar (no Acemap / DashScope calls)."""
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8")

results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((name, True, detail or ""))
    except Exception as exc:  # noqa: BLE001
        results.append((name, False, f"{type(exc).__name__}: {exc}"))
        traceback.print_exc()


# 1. Module imports (heavy deps: gradio, chromadb, sentence-transformers)
def t_imports():
    import rag_system  # noqa: F401
    import rag_generation  # noqa: F401
    import evidence_index  # noqa: F401
    import intent_parser  # noqa: F401
    import gradio_app  # noqa: F401
    return "rag_system, rag_generation, evidence_index, intent_parser, gradio_app"


check("imports", t_imports)


# 2. Interest choices come from community_meta.csv via GraphContext
def t_interests():
    from rag_system import interest_choices
    choices = interest_choices()
    assert len(choices) > 0, "no interest choices"
    return f"{len(choices)} choices, e.g. {choices[0]}"


check("interest_choices", t_interests)


# 3. Input validation happens before any network call
def t_validation():
    from rag_system import enhanced_retrieve, SearchError
    errors = []
    for kwargs, label in [
        ({"question": "   "}, "empty question"),
        ({"question": "x", "interests": [1, 2, 3, 4]}, "too many interests"),
        ({"question": "x", "interests": [99999]}, "invalid interest id"),
    ]:
        try:
            enhanced_retrieve(kwargs.pop("question"), **kwargs)
            errors.append(f"{label}: no error raised")
        except SearchError:
            pass
    assert not errors, errors
    return "3 validation cases rejected as expected"


check("validation", t_validation)


# 4. Full pipeline with replayed (fake) Acemap responses and no LLM.
#    The LLM calls must fail cleanly into a GenerationError path, not crash.
def t_offline_pipeline():
    from rag_system import enhanced_retrieve

    def empty_source(term, size, *, order_by=None):
        return []

    result = enhanced_retrieve(
        "近五年板块构造有哪些研究进展？", mode="expert", paper_source=empty_source
    )
    for key in ("mode", "steps", "conclusion", "papers", "generation_status", "retrieval"):
        assert key in result, f"missing key {key}"
    assert len(result["steps"]) == 4, "steps must be fixed at 4"
    return f"status={result['generation_status']} steps={[s['stage'] for s in result['steps']]}"


check("offline_pipeline_no_results", t_offline_pipeline)


# 5. Prompt templates render with the four fixed placeholders
def t_prompts():
    from modes import resolve
    from pathlib import Path
    root = Path("prompts")
    for mode in ("novice", "intermediate", "expert"):
        cfg = resolve(mode)
        text = (Path(cfg["prompt_file"])).read_text(encoding="utf-8")
        rendered = text.format_map(
            {"question": "Q", "profile": cfg["profile"], "papers": "P", "glossary": "G"}
        )
        assert "{question}" not in rendered and "{papers}" not in rendered, "placeholder left unrendered"
        assert "Q" in rendered and "P" in rendered, "placeholder values missing"
    return "3 prompts render with fixed placeholders"


check("prompt_templates", t_prompts)


# 6. Chinese-topic detection and paper type hint (pure logic, no network)
def t_intent():
    from intent_parser import parse_intent
    intent = parse_intent("近五年板块构造有哪些研究进展？", current_year=2026)
    return f"topic={intent.topic!r} year_range={intent.year_range} type={intent.paper_type}"


check("intent_parser", t_intent)


print()
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
print()
failed = [name for name, ok, _ in results if not ok]
print(f"TOTAL {len(results)}  PASS {len(results) - len(failed)}  FAIL {len(failed)}")
sys.exit(1 if failed else 0)
