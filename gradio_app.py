"""Local Gradio interface for GraphRAG4Scholar."""

import logging
import re
from html import escape, unescape
from pathlib import Path
from urllib.parse import urlsplit

import gradio as gr

from modes import DEFAULT_MODE
from rag_system import MAX_INTERESTS, SearchError, enhanced_retrieve, interest_choices


MODE_CHOICES = (
    ("novice", "初学者", "侧重关键概念与背景。"),
    ("intermediate", "进阶", "兼顾近期进展和经典文献。"),
    ("expert", "专家", "优先近期论文与技术细节。"),
)
EXAMPLES = (
    ("板块构造近五年", "近五年板块构造有哪些研究进展？"),
    ("量子计算进展", "量子计算有哪些研究进展？"),
)
LOGGER = logging.getLogger(__name__)
REFERENCE = re.compile(r"\[(\d+)\]")
TITLE_FORMATTING = re.compile(r"</?(?:scp|italic|bold|sup|sub|i|b|em|strong)\b[^<>]*>", re.I)

ROOT = Path(__file__).resolve().parent
CSS = (ROOT / "gradio_styles.css").read_text(encoding="utf-8")
WORKBENCH_JS = (ROOT / "gradio_workbench.js").read_text(encoding="utf-8")
ICONS = {path.stem: path.read_text(encoding="utf-8") for path in (ROOT / "gradio_icons").glob("*.svg")}


def _icon(name):
    return f'<span class="scholar-icon" aria-hidden="true">{ICONS[name]}</span>'


def _header():
    return (
        '<header class="scholar-header">'
        f'<div class="scholar-brand">{_icon("sitemap")}<div class="scholar-brand-copy">'
        '<strong translate="no">GraphRAG4Scholar</strong></div></div>'
        '<p class="scholar-source">文献来源 <a href="https://acemap.info/" target="_blank" '
        'rel="noopener noreferrer" translate="no">Acemap</a></p></header>'
    )


def _filters():
    options = "".join(
        f'<option value="{value}"{" selected" if value == DEFAULT_MODE else ""}>{label}</option>'
        for value, label, _ in MODE_CHOICES
    )
    segments = "".join(
        f'<button type="button" class="scholar-mode-option" data-mode="{value}" '
        f'title="{help_text}" aria-pressed="{str(value == DEFAULT_MODE).lower()}">{label}</button>'
        for value, label, help_text in MODE_CHOICES
    )
    interests = "".join(
        f'<label><input type="checkbox" name="interest" value="{cid}" '
        f'data-label="{escape(label, quote=True)}"><span>{escape(label)}</span></label>'
        for label, cid in interest_choices()
    )
    return (
        '<section class="scholar-filter-bar" aria-label="检索设置"><div class="scholar-filter-mode">'
        '<label class="scholar-field-label" id="mode-label" for="reader-mode">检索模式</label>'
        f'<select id="reader-mode" class="scholar-mode-select">{options}</select>'
        f'<div class="scholar-mode-options" role="group" aria-labelledby="mode-label">{segments}</div></div>'
        f'<div class="scholar-interest"><p class="scholar-field-label">兴趣领域 <span>（最多 {MAX_INTERESTS} 项）</span></p>'
        '<div id="interest-tags" class="scholar-selected"></div>'
        f'<details id="interest-panel" data-max-interests="{MAX_INTERESTS}">'
        f'<summary>选择兴趣领域{_icon("chevron-down")}</summary>'
        f'<div class="scholar-interest-list">{interests}</div></details>'
        '<p id="interest-notice" role="status"></p></div></section>'
    )


def _composer():
    return (
        '<form id="scholar-search" class="scholar-search" role="search">'
        f'<div class="scholar-query">{_icon("search")}'
        '<input type="search" id="scholar-question" name="question" aria-label="科研问题" '
        'placeholder="输入一个科研问题…" autocomplete="off" required>'
        f'<button type="button" id="clear-question" aria-label="清空问题" hidden>{_icon("x")}</button></div>'
        f'<button type="submit" id="search-button" class="scholar-submit">{_icon("search")}<span>搜索</span></button></form>'
    )


def _empty_result():
    examples = "".join(
        f'<button type="button" data-example="{escape(question, quote=True)}">{label}</button>'
        for label, question in EXAMPLES
    )
    return (
        f'<section class="scholar-empty"><h2>{_icon("book")}<span>示例问题</span></h2>'
        f'<div class="scholar-examples" aria-label="示例问题">{examples}</div></section>'
    )


def _paper_title(paper):
    plain_title = TITLE_FORMATTING.sub("", unescape(str(paper["title"])))
    title = escape(" ".join(plain_title.split()))
    url = paper.get("url") or ""
    if not isinstance(url, str):
        return title
    try:
        parsed = urlsplit(url)
        safe = parsed.scheme in ("https", "http") and parsed.hostname and parsed.username is None
    except ValueError:
        safe = False
    if safe:
        return f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
    return title


def _render_result(result):
    papers = []
    reference_numbers = set()
    for paper in result["papers"]:
        citation = str(int(paper["citation"]))
        reference_numbers.add(citation)
        year = escape(str(paper.get("year") or "年份未收录"))
        cited = paper.get("cited_by_count")
        cited_meta = f'<span class="scholar-cited">被引 {cited:,} 次</span>' if type(cited) is int and cited > 0 else ""
        badge = '<span class="scholar-recommended">推荐</span>' if paper.get("recommended") else ""
        context = (paper.get("context") or "").strip()
        excerpt = ""
        if context:
            excerpt = (
                '<details class="scholar-excerpt"><summary aria-label="查看摘要片段">'
                '<span class="scholar-excerpt-label">摘要</span>'
                f'<span class="scholar-excerpt-preview">{escape(context)}</span></summary>'
                f'<p>{escape(context)}</p></details>'
            )
        papers.append(
            f'<li id="paper-{citation}"><span class="scholar-ref">[{citation}]</span><div>'
            f'<div class="scholar-paper-title">{_paper_title(paper)}</div>'
            f'<p class="scholar-paper-meta">{year}{cited_meta}{badge}</p>{excerpt}</div></li>'
        )

    def link_reference(match):
        number = match[1]
        if number not in reference_numbers:
            return match[0]
        return f'<a class="scholar-cite-link" href="#paper-{number}" aria-label="跳到论文 {number}">[{number}]</a>'

    answer = "".join(
        f'<p class="scholar-answer">{REFERENCE.sub(link_reference, escape(part))}</p>'
        for part in re.split(r"(?<=。)\s*", result["conclusion"]) if part.strip()
    )
    warnings = "".join(_status(str(message), attention=True) for message in result.get("warnings", []))
    citations = f'<ol class="scholar-papers">{"".join(papers)}</ol>' if papers else "<p>未找到可引用的论文。</p>"
    return (
        f'<section class="scholar-results" aria-label="检索结果">{warnings}'
        '<section class="scholar-card scholar-conclusion">'
        f'<header class="scholar-card-header"><h2>学术综合回答</h2></header>{answer}</section>'
        '<section class="scholar-card scholar-citations">'
        f'<header class="scholar-card-header"><h2>主要参考文献 <span>（{len(papers)} 篇）</span></h2></header>'
        f'{citations}</section></section>'
    )


def _render_trace(result=None):
    header = '<div class="scholar-trace-card"><header class="scholar-trace-heading"><h2>检索溯源</h2>'
    if result is None:
        return f'{header}</header><p class="scholar-trace-empty">提交问题后，查看实际检索步骤。</p></div>'

    mode = escape(result["mode"].title())
    steps = "".join(
        f'<li><span class="scholar-step-number" aria-hidden="true">{index}</span>'
        f'<div><h3>{escape(step["title"])}</h3><p>{escape(step["detail"])}</p></div></li>'
        for index, step in enumerate(result["steps"], start=1)
    )
    return f'{header}<p class="scholar-mode">Mode: {mode}</p></header><ol class="scholar-steps">{steps}</ol></div>'


def _status(message, *, attention=False, pending=False):
    css_class = "scholar-status attention" if attention else "scholar-status"
    marker = ' data-search-pending="true"' if pending else ""
    return f'<p class="{css_class}" role="status"{marker}>{escape(message)}</p>'


def _search(event: gr.EventData):
    question = event.question.strip() if isinstance(event.question, str) else ""
    interests = event.interests if isinstance(event.interests, list) else []
    empty_trace = _render_trace()
    if not question:
        yield _status("请输入科研问题。", attention=True), empty_trace
        return
    if len(interests) > MAX_INTERESTS or any(type(cid) is not int for cid in interests):
        yield _status(f"请选择最多 {MAX_INTERESTS} 个有效的兴趣领域。", attention=True), empty_trace
        return
    yield _status("正在检索论文并生成结论…", pending=True), empty_trace
    try:
        result = enhanced_retrieve(question, event.mode, interests)
        yield _render_result(result), _render_trace(result)
    except SearchError as error:
        yield _status(str(error), attention=True), empty_trace
    except Exception:
        LOGGER.exception("Gradio search failed")
        yield _status("检索失败。请查看本地终端日志后重试。", attention=True), empty_trace


def build_app():
    """Build the Blocks app without launching a server."""
    with gr.Blocks(title="GraphRAG4Scholar", fill_width=True, analytics_enabled=False) as app:
        with gr.Row(elem_id="scholar-workspace"):
            gr.HTML(_header(), elem_id="scholar-header", apply_default_css=False, js_on_load=None)
            composer = gr.HTML(_composer(), elem_id="scholar-composer", apply_default_css=False, js_on_load=WORKBENCH_JS)
            gr.HTML(_filters(), elem_id="scholar-filters", apply_default_css=False, js_on_load=None)
            with gr.Column(elem_id="scholar-main"):
                result = gr.HTML(_empty_result(), elem_id="scholar-result", apply_default_css=False, js_on_load=None)
            with gr.Column(elem_id="scholar-trace"):
                trace = gr.HTML(_render_trace(), apply_default_css=False, js_on_load=None)
        composer.submit(_search, inputs=[], outputs=[result, trace], api_visibility="private", show_progress="hidden")
    return app.queue()


def main():
    build_app().launch(
        server_name="127.0.0.1",
        share=False,
        css=CSS,
        theme=gr.themes.Base(font=["DM Sans", "Microsoft YaHei", "sans-serif"]),
        footer_links=[],
    )


if __name__ == "__main__":
    main()
