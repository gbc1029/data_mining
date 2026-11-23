"""
模块A 公共工具：图加载与规范化。

约定的口径（模块B 也必须按这个来）：
    - 节点 key 统一小写去多余空格；
    - 原始写法存在节点属性 display_name 里；
    - 边权仍是共现次数（原始值），归一化权重留给社区检测那一步现算。

CLI 用法：
    python graph_utils.py concept_edges.csv
"""

import csv
import re
import sys
from collections import Counter, defaultdict

import networkx as nx


def norm_name(name):
    """统一节点写法：小写 + 合并空白。"""
    if not name:
        return ""
    return re.sub(r"\s+", " ", name).strip().lower()


def load_nodes(nodes_csv):
    """读节点表，返回 {norm_name: {display_name, level, paper_count, wikidata}}。"""
    meta = {}
    with open(nodes_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = norm_name(row.get("concept", ""))
            if not key:
                continue
            meta[key] = {
                "display_name": row.get("concept", "").strip(),
                "level": row.get("level", ""),
                "paper_count": int(row["paper_count"]) if row.get("paper_count") else 0,
                "wikidata": row.get("wikidata", ""),
            }
    return meta


def load_graph(edges_csv, nodes_csv=None, min_weight=3, drop_isolated=True):
    """
    读边表，返回 nx.Graph。

    min_weight: 低于该共现次数的边丢弃（噪声边）
    drop_isolated: 丢弃度数为 0 的节点（否则 Louvain 会把它们各分成一个社区）
    """
    G = nx.Graph()
    kept = dropped = 0

    with open(edges_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_a = row.get("source", "").strip()
            raw_b = row.get("target", "").strip()
            a, b = norm_name(raw_a), norm_name(raw_b)
            if not a or not b or a == b:
                continue
            w = int(row.get("weight") or 0)
            if w < min_weight:
                dropped += 1
                continue

            # 同名节点合并时保留第一次出现的大小写写法
            if a not in G:
                G.add_node(a, display_name=raw_a)
            if b not in G:
                G.add_node(b, display_name=raw_b)
            G.add_edge(a, b, weight=w)
            kept += 1

    if nodes_csv:
        for key, info in load_nodes(nodes_csv).items():
            if key in G:
                G.nodes[key]["display_name"] = info["display_name"]
                G.nodes[key]["level"] = info["level"]
                G.nodes[key]["paper_count"] = info["paper_count"]

    n_before = G.number_of_nodes()
    if drop_isolated:
        isolated = [n for n, d in G.degree() if d == 0]
        G.remove_nodes_from(isolated)
    else:
        isolated = []

    print(f"[加载] 边 {kept} 条（丢弃 weight<{min_weight}: {dropped} 条）")
    print(f"[加载] 节点 {n_before} 个（丢弃孤立节点 {len(isolated)} 个）→ 保留 {G.number_of_nodes()} 个")
    return G


def prune_hubs(G, top_k=None, ratio=None):
    """
    剔除超级节点。这类概念几乎和所有节点相连，会把社区结构压成一团。
    两种口径二选一：按度数排名取前 top_k 个，或度数超过 ratio*n 的全部剔除。
    """
    n = G.number_of_nodes()
    if top_k:
        hubs = [v for v, _ in Counter(dict(G.degree())).most_common(top_k)]
    elif ratio:
        hubs = [v for v, d in G.degree() if d > ratio * n]
    else:
        return G, []

    G2 = G.copy()
    G2.remove_nodes_from(hubs)
    print(f"[剪枝] 剔除超级节点 {len(hubs)} 个: {[G.nodes[h]['display_name'] for h in hubs]}")
    print(f"[剪枝] 剩余 {G2.number_of_nodes()} 节点 / {G2.number_of_edges()} 边")
    return G2, hubs


def graph_stats(G):
    """打印图的基本统计，Day1 用来核对数据。"""
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        print("[统计] 空图")
        return {}

    deg = [d for _, d in G.degree()]
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    weights = [d["weight"] for _, _, d in G.edges(data=True)]

    stats = {
        "nodes": n,
        "edges": m,
        "density": nx.density(G),
        "avg_degree": 2 * m / n,
        "max_degree": max(deg),
        "components": len(comps),
        "largest_component": comps[0],
        "weight_min": min(weights),
        "weight_max": max(weights),
        "weight_median": sorted(weights)[len(weights) // 2],
    }

    print("\n[统计]")
    print(f"  节点 {n}  边 {m}")
    print(f"  密度 {stats['density']:.4f}  平均度 {stats['avg_degree']:.2f}  最大度 {stats['max_degree']}")
    print(f"  连通分量 {stats['components']} 个，最大分量 {stats['largest_component']} 个节点")
    print(f"  边权 min/median/max = {stats['weight_min']}/{stats['weight_median']}/{stats['weight_max']}")

    print("\n[度数 Top10]")
    for name, d in Counter(dict(G.degree())).most_common(10):
        print(f"  {G.nodes[name]['display_name']:45s} degree={d}")
    return stats


def main():
    edges_csv = sys.argv[1] if len(sys.argv) > 1 else "concept_edges.csv"
    nodes_csv = sys.argv[2] if len(sys.argv) > 2 else None

    G = load_graph(edges_csv, nodes_csv=nodes_csv, min_weight=3)
    graph_stats(G)


if __name__ == "__main__":
    main()
