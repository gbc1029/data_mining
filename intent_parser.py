"""Extract a search topic and explicit paper constraints from a question."""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Intent:
    topic: str
    year_range: tuple[int, int] | None = None
    paper_type: str | None = None
    search_query: str | None = None
    needs_topic_resolution: bool = False
    is_overview: bool = False


_YEAR = r"(?:18|19|20|21)\d{2}"
_YEAR_RANGE = re.compile(
    rf"(?:\b(?:from|between|in|during)\s+|[自从在于]\s*)?(?<!\d)({_YEAR})\s*年?\s*"
    rf"(?:[-–—/]|至|到|\b(?:to|through|and)\b)\s*({_YEAR})(?!\d)\s*年?",
    re.IGNORECASE,
)
_YEAR_BOUND = re.compile(
    rf"\b(since|after|before|until|through|in)\s+({_YEAR})\b"
    rf"|(?:[自从在于]\s*)?(?<!\d)({_YEAR})\s*年\s*(及以后|以来|以后|之后|以前|之前|起|后|前)?",
    re.IGNORECASE,
)
_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_RECENT_YEARS = re.compile(
    r"\b(?:(?:in|over|during|within|from)\s+)?(?:the\s+)?"
    r"(?:last|past|previous)\s+(\d{1,2}|[a-z]+)\s+years?\b"
    r"|(?:在\s*)?(?:最近|过去|近)\s*(\d{1,2}|[一二两三四五六七八九十])\s*年",
    re.IGNORECASE,
)
_THIS_OR_LAST_YEAR = re.compile(
    r"\b(?:(?:in|from)\s+)?(this|last)\s+year\b|(?:在\s*)?(今年|去年)", re.IGNORECASE
)
_PAPER_TYPES = (
    ("meta_analysis", re.compile(r"\bmeta[- ]analys(?:is|es)\b|荟萃分析|元分析", re.I)),
    ("survey", re.compile(
        r"\bsurveys?\s+(?:papers?|on|of|about)\b|\b(?:systematic|literature)\s+reviews?\b"
        r"|\breviews?\s+(?:articles?|papers?|of|on|about)\b|综述|回顾性综述",
        re.I,
    )),
    ("tutorial", re.compile(r"\btutorials?\b|教程|入门资料", re.I)),
    ("preprint", re.compile(r"\bpreprints?\b|预印本", re.I)),
)
_ENGLISH_REQUEST = re.compile(
    r"^(?:(?:please|can you|could you)\s+)?"
    r"(?:what (?:is|are|were)|how (?:does|do|can)|find me|find|show(?: me)?|"
    r"give(?: me)?|provide(?: me)?(?: with)?|summari[sz]e|describe|"
    r"recommend|list|tell me about|explain|compare)\s+"
    r"(?:(?:some|the|a|an)\s+)?",
    re.IGNORECASE,
)
_ENGLISH_TOPIC_PREFIX = re.compile(
    r"^(?:(?:latest|recent|new|some|best|top|notable|seminal|classic|influential)\s+)*"
    r"(?:(?:systematic|literature|research)\s+)?"
    r"(?:advances?|developments?|progress|work|research(?:\s+papers?)?|papers?|"
    r"surveys?(?:\s+papers?)?|review(?:s|\s+(?:articles?|papers?))?|"
    r"tutorials?|preprints?|meta[- ]analys(?:is|es)|research\s+trends?)\s+"
    r"(?:about|on|of|for|in|related to)\s+",
    re.IGNORECASE,
)
_OVERVIEW = re.compile(
    r"(?:近况|现状|研究进展|研究趋势|发展到(?:哪一步|什么阶段|哪了)|"
    r"(?:最新|近期|当前)(?:发展|进展)|现在(?:是)?什么情况|目前(?:是)?什么情况)"
    r"|\b(?:overview|(?:current|present)\s+(?:state|status)|state[- ]of[- ]the[- ]art|"
    r"(?:state|status)\s+of|(?:progress|advances?|developments?)\s+(?:in|on|of)|"
    r"(?:recent|latest|research)\s+(?:progress|advances?|developments?|trends?))\b",
    re.I,
)
_OVERVIEW_PREFIX = re.compile(
    r"^(?:(?:the|a|an)\s+)?(?:(?:current|present|recent|latest)\s+)?"
    r"(?:overview|state(?:[- ]of[- ]the[- ]art)?|status)\s+(?:of|in|on)\s+",
    re.I,
)
_OVERVIEW_SUFFIX = re.compile(
    r"\s*(?:的)?(?:最新|近期|当前|目前|现在)?(?:近况|研究现状|现状|研究进展|研究趋势|"
    r"进展|发展到(?:哪一步|什么阶段|哪了))(?:是?(?:怎么样|如何))?(?:了|吗|呢)?$",
)
_SCOPED_TOPIC = re.compile(
    r"\b(?:for|in|on|under|with|versus|vs|compared (?:to|with))\b"
    r"|(?:用于|应用于|针对|对比|比较|在.+中)",
    re.I,
)
_CHINESE_PREFIX = re.compile(
    r"^(?:请(?:问)?[，, ]*)?(?:(?:给我|帮我)?(?:介绍(?:一下)?|推荐|查找|找一下|总结)"
    r"|什么是|我想了解|如何理解)\s*(?:(?:一下|一些)\s*)?(?:关于|有关)?\s*"
    r"|^(?:关于|有关)\s*"
)
_CHINESE_SUFFIX = re.compile(
    r"\s*(?:是(?:什么|怎么回事)|如何工作|怎么用)$"
    r"|\s*(?:(?:现在|目前)\s*是?\s*(?:什么情况|怎么样|进展如何)"
    r"|是?\s*(?:怎么|如何)\s*(?:做|实现|工作)的?)\s*(?:了|吗|呢)?$"
    r"|\s*(?:综述(?:论文)?|研究进展)有(?:哪些|什么)\s*(?:吗|呢)?$"
    r"|\s*(?:的|有(?:哪些|什么)|(?:有哪些|有什么)的)\s*"
    r"(?:最新|近期|近几年|当前|经典|主要|相关|值得读的)*\s*"
    r"(?:研究进展|研究趋势|进展|新进展|综述(?:论文)?|教程|预印本|"
    r"(?:研究)?论文|文献|方法|应用|研究现状)(?:有(?:哪些|什么))?\s*(?:吗|呢)?$"
)
_CLAUSE_BREAK = re.compile(
    r"[?!。;]+\s*|\.\s+(?=(?:how|what|why|which|where|when|can|does)\b)", re.I
)
_ENGLISH_QUESTION = re.compile(
    r"^(?:how|what|why|which|who|whom|whose|where|when|can|could|would|should|"
    r"does|do|did|is|are|was|were|will)\b", re.I
)
_QUESTION_RESIDUE = re.compile(
    r"^(?:i (?:want|need|would like|wonder)|does anyone|differences? between)\b"
    r"|\b(?:can|could|should|would|will|must)\s+[a-z]+\b"
    r"|(?:怎么|如何|哪些|什么|为什么|是否|有没有|能否|能不能)",
    re.I,
)
_FOLLOW_UP = re.compile(
    r"^(?:(?:and|then|also)\s+|(?:那么|另外|还有)[,\s]*)?"
    r"(?:(?:(?:how|what|why)(?:\s+well)?\s+)?(?:is|are|do|does|did|can|could)\s+"
    r"(?:it|its|they|their|this|that|these|those)\b"
    r"|what about\s+(?:it|them|its|their|this|that)\b"
    r"|它(?:们)?|其(?:作用|局限|优点|缺点|机制|如何|怎么)|(?:该|这种|这项)(?:方法|技术|模型|框架|系统))",
    re.I,
)
_NO_TOPIC = re.compile(
    r"(?:(?:recent|latest|relevant|good|some)\s+)*(?:papers?|articles?|surveys?|reviews?|research)"
    r"|(?:please\s+)?(?:find|explain|compare|recommend)"
    r"|(?:(?:有哪些|有什么|请找|找|推荐)\s*)?(?:(?:最新|近期|高质量|代表性|经典)\s*)*"
    r"(?:论文|综述(?:论文)?|文献|教程|预印本)(?:吗|呢)?",
    re.I,
)
_HAN = re.compile(r"[\u3400-\u9fff]")
_TOPIC_ALIASES = {
    "板块构造": "plate tectonics",
    "板块构造理论": "plate tectonics",
    "图神经网络": "graph neural network",
    "大语言模型对齐": "LLM alignment",
    "大模型对齐": "LLM alignment",
    "量子计算": "quantum computing",
    "气候变化": "climate change",
}


def _year_window(text: str, year: int) -> tuple[tuple[int, int] | None, str]:
    """Find an explicit window and remove only that phrase from the search text."""
    match = _YEAR_RANGE.search(text)
    if match:
        first, last = map(int, match.groups())
        return (min(first, last), max(first, last)), _without(text, match)

    match = _RECENT_YEARS.search(text)
    if match:
        count = (match[1] or match[2]).lower()
        years = int(count) if count.isdecimal() else _NUMBERS.get(count)
        if years:
            return (year - years + 1, year), _without(text, match)

    match = _THIS_OR_LAST_YEAR.search(text)
    if match:
        target = year - int((match[1] or match[2]).lower() in {"last", "去年"})
        return (target, target), _without(text, match)

    match = _YEAR_BOUND.search(text)
    if match:
        operator = (match[1] or match[4] or "").lower()
        boundary = int(match[2] or match[3])
        if operator in {"after", "以后", "之后", "后"}:
            window = (boundary + 1, year)
        elif operator in {"since", "及以后", "以来", "起"}:
            window = (boundary, year)
        elif operator in {"before", "以前", "之前", "前"}:
            window = (1, boundary - 1)
        elif operator in {"through", "until"}:
            window = (1, boundary)
        else:
            window = (boundary, boundary)
        return window, _without(text, match)

    return None, text


def _without(text: str, match: re.Match) -> str:
    before = re.sub(r"\b(?:published|released|appearing)\s*$", "", text[:match.start()], flags=re.I)
    return " ".join(f"{before} {text[match.end():]}".split())


def _topic_query(topic: str, *, has_year: bool) -> tuple[str, str | None, bool]:
    how_question = topic.lower().startswith(("how does ", "how do ", "how can "))
    request = _ENGLISH_REQUEST.match(topic)
    unresolved = request is None and bool(_ENGLISH_QUESTION.match(topic))
    if request:
        topic = topic[request.end():]
    topic = _ENGLISH_TOPIC_PREFIX.sub("", topic)
    topic = _OVERVIEW_PREFIX.sub("", topic)
    search_query = None
    if how_question:
        clause = re.fullmatch(r"(.+?)\s+(work|differ)(?:\s+(.+))?", topic, re.IGNORECASE)
        unresolved = clause is None
        if clause:
            topic, verb, qualifier = clause.groups()
            if qualifier and re.match(r"(?:for|from|in|on|with|under)\s+", qualifier, re.IGNORECASE):
                if verb.lower() == "differ":
                    qualifier = re.sub(r"^from\s+", "versus ", qualifier, flags=re.IGNORECASE)
                search_query = f"{topic} {qualifier}"
            elif qualifier and qualifier.casefold() != "exactly":
                unresolved = True
    topic = _CHINESE_PREFIX.sub("", topic)
    if has_year:
        topic = topic.removeprefix("的").lstrip()
    topic = re.sub(r"^(?:近年来|近几年|最近几年|最近|最新)\s*", "", topic)
    topic = _CHINESE_SUFFIX.sub("", topic)
    topic = _OVERVIEW_SUFFIX.sub("", topic)
    topic = topic.strip(" ,，:：;；\"“”「」")
    return _TOPIC_ALIASES.get(topic, topic), search_query, unresolved or bool(_QUESTION_RESIDUE.search(topic))


def _contains_topic(focused: str, candidate: str) -> bool:
    if _HAN.search(candidate):
        return candidate in focused
    return bool(re.search(rf"(?<![a-z0-9_]){re.escape(candidate)}(?![a-z0-9_])", focused))


def _suspected_abbreviation(topic: str) -> bool:
    tokens = topic.split()
    if any(re.fullmatch(r"[A-Z]{2,8}|[A-Za-z]+\d[A-Za-z0-9]*", token) for token in tokens):
        return True
    # Short initialisms cover colloquial phrases while leaving full words intact.
    return len(tokens) > 1 and bool(re.fullmatch(r"[bcdfghjklmnpqrstvwxz]{2,3}|[aeiou]{2}", tokens[0], re.I))


def parse_intent(question: str, *, current_year: int | None = None) -> Intent:
    """Extract a topic and constraints, plus a focused term when available."""
    text = " ".join(unicodedata.normalize("NFKC", question).split())
    text = text.rstrip(" ?!？！，。;；.")
    if not text:
        raise ValueError("请输入科研问题。")

    year_range, search_text = _year_window(text, current_year or date.today().year)
    paper_type = next((kind for kind, pattern in _PAPER_TYPES if pattern.search(search_text)), None)
    is_overview = bool(_OVERVIEW.search(search_text))
    candidates = []
    for clause in _CLAUSE_BREAK.split(search_text):
        clause = clause.strip()
        if _FOLLOW_UP.match(clause):
            continue
        topic, query, unresolved = _topic_query(clause, has_year=year_range is not None)
        if topic and not _NO_TOPIC.fullmatch(topic):
            candidates.append((topic, query, unresolved))

    if not candidates:
        return Intent(text, year_range, paper_type, needs_topic_resolution=True, is_overview=is_overview)

    topic, query, _ = max(candidates, key=lambda candidate: (len(candidate[0]), candidate[1] is not None))
    unresolved = any(
        pending or not _contains_topic(topic.casefold(), candidate.casefold())
        or (qualifier and qualifier.casefold() != (query or "").casefold())
        for candidate, qualifier, pending in candidates
    )
    scoped = query or _SCOPED_TOPIC.search(topic) or re.search(r"\bcompare\b", search_text, re.I)
    if not scoped:
        unresolved |= _suspected_abbreviation(topic) or (year_range is None and is_overview)
    return Intent(topic, year_range, paper_type, query, unresolved, is_overview)
