from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import cache, lru_cache
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from urllib.parse import quote, urlsplit
import re

import requests

import graph_api
from AcemapPilot.acemap import fetch
from evidence_index import get_index
from intent_parser import parse_intent
from modes import DEFAULT_MODE, resolve
from rag_generation import (
    INSUFFICIENT_EVIDENCE, GenerationError, generate_conclusion, resolve_search_topic,
    select_relevant_evidence, translate_topic,
)


ROOT = Path(__file__).resolve().parent
PAGE_SIZE = 50
MIN_PAPERS = 3
MAX_INTERESTS = 3
RELIABLE_MATCH = {"exact", "singular"}
TYPE_HINTS = {
    "survey": ("survey", "review", "综述"),
    "tutorial": ("tutorial", "primer", "教程"),
    "preprint": ("preprint", "arxiv"),
    "meta_analysis": ("meta-analysis", "meta analysis", "荟萃分析"),
}
SHANGHAI = timezone(timedelta(hours=8))
REQUEST_LOCK = Lock()
last_request_started = 0.0


class SearchError(RuntimeError):
    """A search error that the application can display directly."""


@cache
def _graph():
    # The A adapter looks up its co-occurrence CSV through this module constant.
    graph_api.EDGES_CSV = str(ROOT / graph_api.EDGES_CSV)
    return graph_api.GraphContext(
        community_pkl=ROOT / graph_api.COMMUNITY_PKL,
        link_pkl=ROOT / graph_api.LINK_PKL,
        meta_csv=ROOT / graph_api.META_CSV,
        glossary_csv=ROOT / graph_api.GLOSSARY_CSV,
    )


@lru_cache(maxsize=8192)
def _locate(term):
    return _graph().locate(term)


def _topic_anchor(topic):
    # Match spelling variants without broadening a phrase to unrelated individual words.
    separator = r"[\s\-‐‑‒–—]+"
    words = re.split(separator, topic.casefold())
    parts = [rf"(?:rag|retrieval{separator}augmented{separator}generation)"
             if word == "rag" else re.escape(word) for word in words]
    if re.fullmatch(r"[a-z]{4,}", words[-1]) and not words[-1].endswith("s"):
        parts[-1] += "s?"
    return re.compile(rf"(?<!\w){separator.join(parts)}(?!\w)", re.IGNORECASE)


def interest_choices():
    graph = _graph()
    communities = graph.communities_by_tier(graph.tier_sequence(DEFAULT_MODE))
    return [(item["label"], item["community_id"])
            for item in sorted(communities, key=lambda item: item["community_id"])]


def _live_papers(term, size, *, order_by=None):
    global last_request_started
    with REQUEST_LOCK:
        sleep(max(0, 1 - (monotonic() - last_request_started)))
        last_request_started = monotonic()
        return fetch(term, size=size, save=False, order_by=order_by)


def _collect(terms, paper_source, order_by):
    papers, failed = {}, []
    for term in terms:
        try:
            rows = paper_source(term, PAGE_SIZE, order_by=order_by)
            if not isinstance(rows, list):
                raise ValueError("Invalid Acemap results")
        except (requests.RequestException, ValueError, KeyError):
            failed.append(term)
            continue
        for row in rows:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            paper_id = str(row["id"])
            existing = papers.setdefault(paper_id, row)
            if not existing.get("abstract") and row.get("abstract"):
                papers[paper_id] = row
    if terms and len(failed) == len(terms):
        raise SearchError("Acemap 暂未返回有效响应，请稍后重试。")
    return list(papers.values()), failed


def _paper(row, use_graph):
    title = row.get("title") or row.get("display_name")
    abstract = row.get("abstract")
    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(abstract, str) or not abstract.strip():
        return None

    tags = {}
    if use_graph:
        # Acemap orders concepts by confidence. Prefer specific, reliable matches.
        for concept in (row.get("concepts") or [])[:6]:
            label = concept.get("display_name")
            if not label or concept.get("level") == 0:
                continue
            info = _locate(label)
            if info and info["match"] in RELIABLE_MATCH:
                tags[info["key"]] = (info["community_id"], info["tier"])

    location = row.get("primary_location") or {}
    url = row.get("doi") or location.get("landing_page_url") or ""
    if url.startswith("10."):
        url = "https://doi.org/" + url
    parsed_url = urlsplit(url)
    return {
        "id": str(row["id"]), "title": title.strip(), "abstract": abstract.strip(),
        "year": row.get("publication_year"), "cited_by_count": row.get("cited_by_count") or 0,
        "url": (url if parsed_url.scheme in {"http", "https"} and parsed_url.netloc
                else f"https://acemap.info/papers/{quote(str(row['id']), safe='')}"),
        "tags": tags,
    }


def _eligible(papers, cfg, topic_communities):
    tiers = list(cfg["tier"])

    def in_scope(paper):
        return any(tier in tiers and (not topic_communities or cid in topic_communities)
                   for cid, tier in paper["tags"].values())

    eligible = [paper for paper in papers if in_scope(paper)]
    changes = []
    for tier in _graph().tier_sequence(cfg["mode"]):
        if len(eligible) >= MIN_PAPERS:
            break
        if tier not in tiers:
            tiers.append(tier)
            eligible = [paper for paper in papers if in_scope(paper)]
            changes.append(tier)
    if len(eligible) < MIN_PAPERS:
        return papers, tiers, changes + ["不限 tier 和社区"]
    return eligible, tiers, changes


def _affinity(paper, interests, links):
    matching = {key for key, (cid, _) in paper["tags"].items() if cid in interests}
    return bool(matching), len(matching & links)


def _evidence_shortlist(papers, passages, cfg, year):
    ids = list(passages)
    if cfg["quota"]:
        # Preserve room to examine both time ranges before applying the final quota.
        published = {paper["id"]: paper["year"] or 0 for paper in papers}
        cutoff = year - 4
        recent = [paper_id for paper_id in ids if published[paper_id] >= cutoff]
        classic = [paper_id for paper_id in ids if 0 < published[paper_id] < cutoff]
        ids = list(dict.fromkeys(recent[:2 * cfg["quota"]["new"]]
                                 + classic[:2 * cfg["quota"]["old"]] + ids))
    return ids[:2 * cfg["top_k"]]


def _rank(papers, passages, cfg, intent, year, baseline, interests, links, anchor):
    by_id = {paper["id"]: paper for paper in papers}
    if baseline:
        return [paper for paper in papers if paper["id"] in passages][:cfg["top_k"]]

    semantic_order = [by_id[paper_id] for paper_id in passages]
    hints = TYPE_HINTS.get(intent.paper_type, ())

    def mentions_topic(paper):
        return bool(anchor.search(paper["title"]) or anchor.search(passages[paper["id"]]))

    def matches_requested_type(paper):
        title_and_url = f"{paper['title']} {paper['url']}".casefold()
        return any(hint in title_and_url for hint in hints)

    def pick(pool, sort, limit):
        # The evidence shortlist already bounds this pool. Keep semantic order on ties.
        metric = (lambda p: p["cited_by_count"]) if sort == "cited" else (lambda p: p["year"] or 0)
        chosen = sorted(
            pool,
            key=lambda p: (mentions_topic(p), matches_requested_type(p), _affinity(p, interests, links), metric(p)),
            reverse=True,
        )[:limit]
        return sorted(chosen, key=metric, reverse=True)

    if cfg["sort"] != "mixed":
        return pick(semantic_order, cfg["sort"], cfg["top_k"])

    recent_since = year - 4
    new = [p for p in semantic_order if (p["year"] or 0) >= recent_since]
    old = [p for p in semantic_order if 0 < (p["year"] or 0) < recent_since]
    chosen = pick(new, "year", cfg["quota"]["new"]) + pick(old, "cited", cfg["quota"]["old"])
    remaining = cfg["top_k"] - len(chosen)
    if not remaining:
        return chosen
    selected_ids = {paper["id"] for paper in chosen}
    overflow = [p for p in semantic_order if p["id"] not in selected_ids]
    return chosen + pick(overflow, "year", remaining)


def _cite(papers, passages, interests, links):
    recommendations = sorted(papers, key=lambda p: _affinity(p, interests, links), reverse=True)[:3] if interests else []
    recommended_ids = {paper["id"] for paper in recommendations if _affinity(paper, interests, links)[0]}

    return [
        {**{key: value for key, value in paper.items() if key != "tags"},
         "citation": number, "context": passages[paper["id"]],
         "community_ids": sorted({cid for cid, _ in paper["tags"].values()}),
         "recommended": paper["id"] in recommended_ids}
        for number, paper in enumerate(papers, start=1)
    ]


def _conclude(question, cfg, papers, terms, *, overview=False):
    evidence = "\n\n".join(
        f"[{p['citation']}] {p['title']} | {p['year'] or '年份未知'} | "
        f"被引 {p['cited_by_count']}\n{p['context']}"
        for p in papers
    )
    hints = _graph().glossary(terms) if cfg["mode"] == "novice" else {}
    template = (ROOT / cfg["prompt_file"]).read_text(encoding="utf-8")
    prompt = template.format_map(defaultdict(str, {
        "question": question,
        "profile": cfg["profile"],
        "papers": evidence,
        "glossary": "\n".join(f"{term}：{hint}" for term, hint in hints.items()),
    }))
    return generate_conclusion(prompt, len(papers), overview=overview)


def enhanced_retrieve(question, mode=DEFAULT_MODE, interests=(), *, baseline=False, paper_source=None):
    """Run one question, optionally replaying paper_source(keyword, size, order_by=...)."""
    question = question.strip()
    if not question:
        raise SearchError("请输入科研问题。")
    interests = set(interests or ())
    if len(interests) > MAX_INTERESTS or not interests <= {cid for _, cid in interest_choices()}:
        raise SearchError(f"请选择最多 {MAX_INTERESTS} 个有效的兴趣领域。")

    cfg = resolve(mode)
    year = datetime.now(SHANGHAI).year
    try:
        intent = parse_intent(question, current_year=year)
    except ValueError as exc:
        raise SearchError(str(exc)) from exc

    year_bounds = [r for r in (intent.year_range, cfg["year_range"]) if r]
    years = (max(r[0] for r in year_bounds), min(r[1] for r in year_bounds)) if year_bounds else None
    conflict = years is not None and years[0] > years[1]

    topic, qualified = intent.topic, intent.search_query
    overview = intent.is_overview
    translate = bool(re.search(r"[\u3400-\u9fff]", topic))
    plan_topic = not conflict and (intent.needs_topic_resolution or (overview and translate))
    try:
        if plan_topic:
            topic, qualified = resolve_search_topic(question, overview=overview)
        elif not conflict and translate:
            topic = translate_topic(topic)
    except GenerationError as exc:
        raise SearchError(f"暂时无法提取检索主题。可以先直接输入研究主题。{exc}") from exc
    if not topic:
        raise SearchError("请在问题中写明研究主题。")

    info = None if baseline else _locate(topic)
    reliable = info and info["match"] in RELIABLE_MATCH
    expansion = _graph().search_terms(topic, cfg["mode"])[1:] if reliable else []
    base = graph_api.clean_term(topic) or topic
    graph_terms = [graph_api.clean_term(term) for term in expansion]
    if baseline:
        search_terms = [base if overview else qualified or base]
    elif overview and plan_topic:
        search_terms = [base, *graph_terms] if graph_terms else [base, qualified]
    else:
        search_terms = [qualified, base, *graph_terms]
    terms = list({term.casefold(): term for term in search_terms if term}.values())
    semantic_query = topic if overview else qualified or (topic if re.search(r"[\u3400-\u9fff]", question) else question)
    anchor = _topic_anchor(base)
    scope = {item["community_id"] for term in [base, *graph_terms] if (item := _locate(term))
             and item["match"] in RELIABLE_MATCH} if reliable and not overview else set()

    warnings = []
    order_by = "publication_date" if years and years[1] >= year - 1 else None
    rows, failed = _collect([] if conflict else terms, paper_source or _live_papers, order_by)
    if conflict:
        warnings.append("问题指定的时间与当前模式没有交集。")
    if failed:
        warnings.append(f"Acemap 有 {len(failed)} 个检索词请求失败，结果使用其余请求的数据。")

    year_rows = [row for row in rows if row.get("publication_year")
                 and years[0] <= row["publication_year"] <= years[1]] if years else rows
    candidates = [paper for row in year_rows if (paper := _paper(row, use_graph=not baseline))]
    passages = get_index().search(semantic_query, candidates) if candidates else {}
    screened_count = 0
    screening_status = "not_run"
    if passages and not baseline:
        shortlist = _evidence_shortlist(candidates, passages, cfg, year)
        screened_count = len(shortlist)
        titles = {paper["id"]: paper["title"] for paper in candidates}
        evidence = [(titles[paper_id], passages[paper_id]) for paper_id in shortlist]
        try:
            selected = select_relevant_evidence(question, topic, evidence, overview=overview)
            accepted = {shortlist[index] for index in selected}
            passages = {paper_id: context for paper_id, context in passages.items() if paper_id in accepted}
            screening_status = "passed"
        except GenerationError as exc:
            passages = {}
            screening_status = "unavailable"
            warnings.append(f"相关性筛选服务暂不可用。{exc}")

    candidates = [paper for paper in candidates if paper["id"] in passages]
    eligible, tiers, changes = candidates, [], []
    if candidates and reliable:
        eligible, tiers, changes = _eligible(candidates, cfg, scope)
        eligible_ids = {paper["id"] for paper in eligible}
        passages = {paper_id: context for paper_id, context in passages.items() if paper_id in eligible_ids}
    related = _graph().expand(topic, mode="expert") if interests and reliable else []
    links = {match["key"] for term in related if (match := _locate(term))}
    ranked = _rank(eligible, passages, cfg, intent, year, baseline, interests, links, anchor)
    papers = _cite(ranked, passages, interests, links)
    generation_status = "insufficient_evidence"
    if papers:
        try:
            conclusion = _conclude(question, cfg, papers, terms, overview=overview)
            if conclusion != INSUFFICIENT_EVIDENCE:
                generation_status = "generated"
        except GenerationError as exc:
            conclusion = "结论暂未生成。可先查看下方检索到的论文与摘要。"
            warnings.append(str(exc))
            generation_status = "unavailable"
    elif screening_status == "unavailable":
        conclusion = "本次论文摘要暂未完成相关性检查，请稍后重试。"
        generation_status = "unavailable"
    else:
        conclusion = "当前检索没有取得能支持这个问题的可用摘要，证据不足以回答。"

    intent_detail = f"主题 {topic} · 时间 {years[0]}–{years[1]}" if years else f"主题 {topic} · 时间不限"
    if conflict:
        intent_detail = f"主题 {topic} · 时间条件冲突"
    if intent.paper_type:
        intent_detail += f" · 类型偏好 {intent.paper_type}"
    if overview:
        intent_detail += " · 研究概况"
    elif qualified:
        intent_detail += f" · 限定检索 {qualified}"
    community_detail = "纯 Acemap 基线" if baseline else "图中没有可靠匹配 · 已转为 Acemap 主题检索"
    if reliable:
        community_detail = f"C{info['community_id']:02d} {info['community_label']}（{info['tier']}）"
    screening_detail = f" → 检查 {screened_count} 篇相关性" if screened_count else ""
    retrieve_detail = f"召回 {len(rows)} 篇{screening_detail} → 保留 {len(papers)} 篇"
    if changes:
        retrieve_detail += " · 已放宽 tier：" + " → ".join([" + ".join(cfg["tier"]), *changes])
    steps = [
        ("intent", "意图解析", intent_detail),
        ("community", "社区定位", community_detail),
        ("expand", "关键词扩展", f"学术检索词 {'、'.join(terms)}"),
        ("retrieve", "检索筛选", retrieve_detail),
    ]
    return {
        "mode": cfg["mode"], "conclusion": conclusion, "papers": papers,
        "steps": [{"stage": stage, "title": title, "detail": " ".join(detail.split())[:240]}
                  for stage, title, detail in steps],
        "degraded": bool(changes), "warnings": warnings, "generation_status": generation_status,
        "contexts": [paper["context"] for paper in papers],
        "retrieval": {"baseline": baseline, "terms": [] if conflict else terms,
                      "interests": sorted(interests), "year_range": years, "tiers": tiers,
                      "relaxations": changes, "recalled": len(rows), "eligible": len(eligible),
                      "failed_terms": failed, "order_by": order_by,
                      "screened": screened_count, "screening_status": screening_status},
    }
