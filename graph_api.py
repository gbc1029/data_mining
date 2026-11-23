"""
模块A → 模块B 的适配层。

B 侧只需要 import 这个类，不用直接碰 pickle / csv；所有 key 归一化、
主题匹配兜底、检索词清洗都在这里完成。

用法：
    from graph_api import GraphContext

    g = GraphContext()
    g.locate("Graph Neural Networks")
    # {'key': 'graph neural network', 'display_name': 'Graph neural network',
    #  'level': 2, 'community_id': 4, 'community_label': '计算机与信息科学', 'tier': 'basic'}

    g.expand("graph neural network", mode="expert")      # 按档位取扩展概念
    g.search_terms("graph neural networks", mode="novice")  # 清洗后的检索词列表
    g.tier_sequence("expert")   # ['subfield', 'niche', 'basic']，B 降级时按序放宽
    g.glossary(["graph neural network", "gene"])         # {term: 白话解释}

命令行自测：
    python graph_api.py
"""

import csv
import pickle

from graph_utils import norm_name
from modes import resolve

COMMUNITY_PKL = "concept_community.pkl"
LINK_PKL = "link_pred_result.pkl"
META_CSV = "community_meta.csv"
GLOSSARY_CSV = "glossary.csv"
EDGES_CSV = "full_concept_edges.csv"

TIER_ORDER = ("basic", "subfield", "niche")   # 由宽到窄


def clean_term(name):
    """去掉维基消歧后缀再归一化：'Set (abstract data type)' -> 'set'。"""
    if not name:
        return ""
    if "(" in name:
        name = name.split("(")[0]
    return norm_name(name)


def _singular_variants(key):
    """只对最后一个词做单复数试探。"""
    head, _, last = key.rpartition(" ")
    out = []
    if last.endswith("ies"):
        out.append(f"{head} {last[:-3]}y".strip())
    if last.endswith(("ses", "xes")):
        out.append(f"{head} {last[:-2]}".strip())
    if last.endswith("s") and not last.endswith("ss"):
        out.append(f"{head} {last[:-1]}".strip())
    return out


def _sing_token(tok):
    """词级单复数归一，用于模糊匹配。"""
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        return tok[:-1]
    return tok


def _tokens(name):
    return [_sing_token(t) for t in clean_term(name).split()]


def _phrase_in(short, long):
    """short 是否作为连续子序列出现在 long 中。"""
    n = len(short)
    return any(long[i:i + n] == short for i in range(len(long) - n + 1))


def _token_match(a, b):
    """同词，或共享 >=5 字符词干：computing ~ computer。"""
    if a == b:
        return True
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= 5


class GraphContext:
    def __init__(self, community_pkl=COMMUNITY_PKL, link_pkl=LINK_PKL,
                 meta_csv=META_CSV, glossary_csv=GLOSSARY_CSV):
        self.community = pickle.load(open(community_pkl, "rb"))
        self.link = pickle.load(open(link_pkl, "rb"))
        self._adj = None

        # 清洗后索引：忽略括号消歧后缀；重名的直接丢弃，避免指错
        self.clean_index = {}
        clash = set()
        for k in self.community:
            ck = clean_term(k)
            if ck in self.clean_index:
                clash.add(ck)
            self.clean_index[ck] = k
        for ck in clash:
            self.clean_index.pop(ck, None)
        self.meta = {}
        self.communities = {}
        with open(meta_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                cid = int(row["community_id"])
                self.meta[cid] = row
                self.communities[cid] = {
                    "community_id": cid,
                    "label": row["label"],
                    "size": int(row["size"]),
                    "tier": row["tier"],
                    "top_concepts": row["top_concepts"],
                }
        self.glossary_map = {}
        with open(glossary_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                self.glossary_map[norm_name(row["term"])] = row["explain"]

    # ---------- 主题定位 ----------

    def match(self, phrase):
        """
        把用户主题映射到图里的概念 key。
        顺序：精确 -> 去括号后精确 -> 单复数 -> 词级模糊。
        返回 (key, 匹配方式)；失败返回 (None, "miss")。
        """
        key = clean_term(phrase)
        if not key:
            return None, "miss"
        if key in self.community:
            return key, "exact"
        if key in self.clean_index:
            return self.clean_index[key], "exact"

        for cand in _singular_variants(key):
            if cand in self.community:
                return cand, "singular"
            if cand in self.clean_index:
                return self.clean_index[cand], "singular"

        t_tokens = _tokens(phrase)
        t_set = set(t_tokens)
        if not t_set:
            return None, "miss"

        best, best_score, best_how = None, None, None
        for k in self.community:
            k_tokens = _tokens(k)
            if not k_tokens:
                continue
            k_set = set(k_tokens)
            hit_topic = {t for t in t_set if any(_token_match(t, x) for x in k_set)}
            hit_key = {x for x in k_set if any(_token_match(x, t) for t in t_set)}
            shared = len(hit_topic)
            if shared == 0:
                continue
            coverage = shared / len(t_set)          # 主题词被覆盖的比例
            key_cov = len(hit_key) / len(k_set)     # 概念词被覆盖的比例
            # 多词查询只共享一个词时，要求对方是多词概念且覆盖过半
            if shared < 2 and not (len(k_set) >= 2 and coverage >= 0.5):
                continue
            score = (shared, coverage, key_cov)
            if best_score is None or score > best_score:
                best, best_score = k, score
                contained = _phrase_in(t_tokens, k_tokens) or _phrase_in(k_tokens, t_tokens)
                best_how = "contains" if contained else "token"

        return (best, best_how) if best else (None, "miss")

    def locate(self, topic):
        """返回主题所在的社区与档位；匹配失败返回 None。"""
        key, how = self.match(topic)
        if not key:
            return None
        info = self.community[key]
        meta = self.meta.get(info["community_id"], {})
        return {
            "key": key,
            "display_name": info["display_name"],
            "level": int(info["level"]),
            "community_id": info["community_id"],
            "community_label": info["community_label"],
            "tier": meta.get("tier"),
            "match": how,
        }

    # ---------- 扩展与检索词 ----------

    def _cooccur(self):
        """共现近邻，按关联强度排序（与社区检测同一口径），首次调用后缓存。"""
        if self._adj is None:
            adj = {}
            with open(EDGES_CSV, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    a = norm_name(row.get("source", ""))
                    b = norm_name(row.get("target", ""))
                    w = int(row.get("weight") or 0)
                    if not a or not b or a == b or w < 3:
                        continue
                    adj.setdefault(a, {})[b] = w
                    adj.setdefault(b, {})[a] = w
            n = len(adj)
            deg = {u: len(nb) for u, nb in adj.items()}
            self._adj = {
                u: sorted(((v, w * n / (deg[u] * deg[v])) for v, w in nb.items()),
                          key=lambda x: -x[1])
                for u, nb in adj.items()
            }
        return self._adj

    def expand(self, topic, mode=None, cfg=None, k=None):
        """按档位取扩展概念。novice/intermediate 用共现近邻，expert 用链路预测。"""
        cfg = cfg or resolve(mode or "intermediate")
        key, _ = self.match(topic)
        if not key:
            return []

        if cfg["expand"] == "link_pred":
            cand = [c for c in self.link.get(key, []) if c in self.community]
        else:
            cand = [v for v, _ in self._cooccur().get(key, []) if v in self.community]

        mode = cfg.get("mode", "intermediate")
        limit = k or (2 if mode == "novice" else 3 if mode == "intermediate" else 5)
        return [self.community[c]["display_name"] for c in cand[:limit]]

    def search_terms(self, topic, mode=None):
        """给 Acemap 用的检索词：清洗后的主题 + 扩展概念，已去重。"""
        key, _ = self.match(topic)
        base = self.community[key]["display_name"] if key else clean_term(topic)
        terms = [base]
        for t in self.expand(topic, mode=mode):
            if t.lower() not in {x.lower() for x in terms}:
                terms.append(t)
        return terms

    # ---------- 档位与社区 ----------

    def tier_sequence(self, mode):
        """降级顺序：先本档 tier，再按宽窄远近补齐其余档。"""
        cfg = resolve(mode)
        seq = list(cfg["tier"])
        anchor = TIER_ORDER.index(seq[0]) if seq and seq[0] in TIER_ORDER else 1
        rest = sorted((t for t in TIER_ORDER if t not in seq),
                      key=lambda t: abs(TIER_ORDER.index(t) - anchor))
        return seq + rest

    def communities_by_tier(self, tiers):
        """给 B 的社区过滤/兴趣勾选：按 tier 取社区列表。"""
        return [c for c in self.communities.values() if c["tier"] in tiers]

    # ---------- 术语表 ----------

    def glossary(self, terms):
        """命中术语表的概念 -> 白话解释；未命中的忽略。"""
        out = {}
        for t in terms:
            key = clean_term(t)
            if key in self.glossary_map:
                out[t] = self.glossary_map[key]
        return out


if __name__ == "__main__":
    g = GraphContext()
    for q in ["Graph Neural Networks", "quantum computing", "Set (abstract data type)"]:
        print(f"[locate] {q!r} -> {g.locate(q)}")
    print("[expand] novice:", g.expand("graph neural network", mode="novice"))
    print("[expand] expert:", g.expand("graph neural network", mode="expert"))
    print("[terms ] expert:", g.search_terms("graph neural networks", mode="expert"))
    for m in ("novice", "intermediate", "expert"):
        print(f"[tier  ] {m}:", g.tier_sequence(m))
    print("[gloss ]", g.glossary(["graph neural network", "gene", "not a concept"]))
