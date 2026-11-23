# 模块A 交付包使用说明（给同学B）

> 本包只包含图挖掘侧产物与适配层。RAG、Gradio、论文拉取仍由你实现。
> 不需要重算任何图算法，也不需要直接读 pickle。

## 1. 包内文件

| 文件 | 用途 |
|------|------|
| `graph_api.py` | 唯一需要 import 的模块 |
| `graph_utils.py` | `norm_name()` 依赖，不要改 |
| `modes.py` | 三档参数 `resolve(mode)` |
| `concept_community.pkl` | 概念 → 社区 / tier（2076 条） |
| `link_pred_result.pkl` | 概念 → Top-5 潜在关联（2076 条） |
| `community_meta.csv` | 16 个社区的 tier 与代表概念 |
| `glossary.csv` | 110 条术语的白话解释 |
| `full_concept_edges.csv` | 共现邻接表，novice/intermediate 扩展要用 |
| `prompts/*.txt` | 三档提示词模板 |

缺 `full_concept_edges.csv` 时，expert 档仍能跑，但 novice/intermediate 的扩展会报 `FileNotFoundError`。

## 2. 三步接入

```bash
python graph_api.py     # 看到 [locate] / [expand] / [tier] 输出即环境正常
python modes.py         # 看到三档 tier/year/sort/top_k
```

```python
from graph_api import GraphContext
from modes import resolve

g = GraphContext()      # 启动时建一次，别在每次请求里 new

cfg     = resolve(mode)               # mode 来自下拉框：novice / intermediate / expert
info    = g.locate(topic)             # None 表示图里没有这个概念
keywords= g.search_terms(topic, mode) # 直接送 Acemap work_search
tiers   = g.tier_sequence(mode)       # 结果不足 3 篇时按序放宽
hints   = g.glossary(keywords) if mode == "novice" else {}
```

## 3. 接口清单

| 方法 | 返回 |
|------|------|
| `g.locate(topic)` | `{key, display_name, level, community_id, community_label, tier, match}` 或 `None` |
| `g.match(phrase)` | `(key, 匹配方式)`，匹配方式为 `exact / singular / contains / token / miss` |
| `g.expand(topic, mode=...)` | 扩展概念列表（novice·intermediate 读共现，expert 读链路预测） |
| `g.search_terms(topic, mode=...)` | 已去重、已剥括号的检索词列表 |
| `g.tier_sequence(mode)` | 降级顺序，如 `['subfield','niche','basic']` |
| `g.communities_by_tier(['niche'])` | 社区列表，可做兴趣勾选框 |
| `g.glossary([...])` | `{术语: 白话解释}`，未命中的自动忽略 |

## 4. 三档行为对照

| | novice | intermediate | expert |
|---|---|---|---|
| tier | basic | basic + subfield | subfield + niche |
| 扩展来源 | 共现近邻 | 共现近邻 | 链路预测 |
| 年份 | 不限 | 不限 | 2021–2026 |
| 排序 | 按被引 | 新旧各半 | 按年份 |
| top_k | 8 | 10 | 12 |

## 5. 必须遵守的约定

1. **不要直接读 pkl / csv**，一律走 `GraphContext`；A 换数据时接口不变。
2. **不要自己清洗概念名**，`search_terms` 已去掉 `(abstract data type)` 这类后缀。
3. **降级顺序用 `tier_sequence`**，不要自己写 tier 列表。
4. 提示词占位符固定四个：`{question}` `{profile}` `{papers}` `{glossary}`，改名会 KeyError。
5. 步骤卡的 `detail` 从 `locate` / `expand` / 召回数里取，不要让大模型编中间步骤。

## 6. `match` 严格度怎么选

- `exact` / `singular`：可靠，可直接用。
- `contains` / `token`：兜底匹配，可能出现跨领域误配，例如
  `protein structure prediction → crystal structure prediction`。
- 建议：检索时接受兜底，但界面上标注「近义匹配」；报告的局限一节需如实说明。

## 7. 常见报错

| 现象 | 原因 |
|------|------|
| `FileNotFoundError: full_concept_edges.csv` | 边表没随包复制 |
| `KeyError: 'mode'` | 自己构造的 cfg 没走 `resolve()` |
| `locate()` 返回 `None` | 图里没有这个概念，前端应提示「未收录」，不要静默报错 |
| `ModuleNotFoundError: networkx` | 环境缺 networkx |

## 8. 变更边界

A 只会在必要时重跑 `build_community_meta.py`（tier 阈值）或补 `glossary.csv`，
**不会修改 `resolve()` 签名和 `GraphContext` 的返回字段**，你可以放心依赖。
