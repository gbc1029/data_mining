"""Use corroborated A-graph edges to admit evidence into a bounded shortlist."""

from collections import Counter
from functools import cache
from math import log1p
import re


def _tokens(text):
    return tuple(re.findall(r"[^\W_]+", text.casefold()))


@cache
def _text_index(graph):
    surfaces = {}
    ambiguous = set()
    for name, key in graph.clean_index.items():
        words = _tokens(name)
        if len(words) < 2 or int(graph.community[key]["level"]) < 2:
            continue
        if words in surfaces and surfaces[words] != key:
            ambiguous.add(words)
        surfaces[words] = key
    for words in ambiguous:
        del surfaces[words]
    return surfaces, sorted({len(words) for words in surfaces})


def _mentioned_concepts(text, graph):
    words = _tokens(text)
    surfaces, lengths = _text_index(graph)
    found = set()
    for length in lengths:
        for i in range(len(words) - length + 1):
            key = surfaces.get(words[i:i + length])
            if key is not None:
                found.add(key)
    return found


def graph_focus(topic, question, graph):
    key, match = graph.match(topic)
    if key and match in {"exact", "singular"}:
        return graph.community[key]["display_name"]
    explicit = _mentioned_concepts(question, graph)
    if len(explicit) == 1:
        return graph.community[next(iter(explicit))]["display_name"]
    return None


def graph_shortlist(order, papers, *, graph, topic_anchor, make_anchor, mode, limit, text_links=False):
    shortlist = order[:limit]
    if len(order) <= limit:
        return shortlist, []

    terms = {}
    for pid in order:
        for key in papers[pid]["tags"]:
            if key not in terms and int(graph.community[key]["level"]) >= 2:
                terms[key] = make_anchor(graph.community[key]["display_name"])
    supported = {}
    for pid in order:
        paper = papers[pid]
        text = f'{paper["title"]}\n{paper["abstract"]}'
        if topic_anchor.search(text):
            supported[pid] = {key for key in paper["tags"] if key in terms and terms[key].search(text)}
            if text_links:
                supported[pid].update(_mentioned_concepts(text, graph))

    votes = Counter(key for pid in order[:10] for key in supported.get(pid, ()))
    seeds = sorted(key for key, count in votes.items() if count >= 2)
    if not seeds:
        return shortlist, []

    frequency = Counter(key for keys in supported.values() for key in keys)
    edges = graph._cooccur()
    bridge = {}
    for seed in seeds:
        neighbors = edges.get(seed, ())
        scale = neighbors[0][1] if neighbors else 1.0
        links = [(key, strength / scale, "cooccurrence") for key, strength in neighbors]
        if mode == "expert":
            links += [(key, 0.5 / position, "predicted")
                      for position, key in enumerate(graph.link.get(seed, ()), 1)]
        for key, strength, relation in links:
            if key in seeds or key not in frequency:
                continue
            score = strength * log1p(len(order) / frequency[key])
            if key not in bridge or score > bridge[key][0]:
                bridge[key] = (score, seed, key, relation)

    def best(pid):
        return max((bridge[key] for key in supported.get(pid, ()) if key in bridge),
                   default=(0.0, "", "", ""))

    alternatives = sorted(order[limit:], key=best, reverse=True)
    tail = sorted(range(max(0, limit - 2), limit), key=lambda i: best(shortlist[i]))
    rescued = []
    for slot, pid in zip(tail, alternatives):
        score, seed, target, relation = best(pid)
        if score <= best(shortlist[slot])[0]:
            break
        rescued.append({"paper_id": pid, "replaced_id": shortlist[slot], "seed_concept": seed,
                        "neighbor_concept": target, "relation": relation,
                        "seed_paper_ids": [head for head in order[:10] if seed in supported.get(head, ())]})
        shortlist[slot] = pid
    return shortlist, rescued
