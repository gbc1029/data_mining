# GraphRAG4Scholar 评估报告（初版）

> 版本：初版 · 生成于本轮整合与试评估之后
> 说明：本文先忠实描述系统功能、原理与运行架构，再记录已完成的验证与仍存在的问题。
> 凡属“已完成”的结论，均指在本工作区、当前环境（`data_mining`）中实际运行得到的结果；
> 凡属推断或未经核对的内容，均在文中明确标注。

---

## 一、项目背景与目标

GraphRAG4Scholar 是一个面向学术问答的检索增强生成（RAG）系统。它针对纯关键词检索的两个不足——**语义相关但关键词不匹配**、**缺乏个性化与用户分层**——引入图数据挖掘结果来增强检索，并提供基于用户兴趣勾选的个性化推荐与三档用户模式。

系统由三名成员分工完成：

| 模块 | 职责 | 主要产物 |
|---|---|---|
| A | 图挖掘预处理 | Acemap 全领域 Concept 共现图、社区检测、链路预测、社区元数据 |
| B | 个性化 RAG 系统 | 检索、排序、生成、三档配置、Gradio 前端 |
| C | 整合、评估、展示 | 测试问题集、评估报告、对比实验、海报与演示视频 |

本文由模块 C 编写，覆盖整体框架与现状；A 的图挖掘方法按其交付文档与协作指南中的记录转述，未重新运行验证。

---

## 二、核心方法与原理

### 2.1 模块 A：图挖掘

A 的数据基础是 Acemap 全领域 Concept 共现图。边权经过度归一化以削弱超级节点的支配作用：

```
w'(u,v) = weight(u,v) × n / (deg(u) × deg(v))
```

其中 `n` 为图中概念数，`deg` 为节点度。

- **社区检测**：Louvain 算法，`resolution=0.6`，`seed=42`。据 A 的记录，得到 16 个社区，模块度 0.801。
- **链路预测**：Adamic-Adar 指数，TopK=5，候选集限定为「同社区 + level ≥ 2 + 非邻居」，每节点最多采样 300 个。评估采用 90/10 划分边并配等量负样本，按秩计算 AUC：AA 0.9699、Jaccard 0.7878、RA 0.9792，P@5=0.0550。
- **社区元数据**：`community_meta.csv` 按社区规模划分 tier：`size ≥ 150 → basic`，`57–149 → subfield`，`< 57 → niche`，实测为 5 / 8 / 3 个社区。

这些数据以 `concept_community.pkl`（概念→社区映射）与 `link_pred_result.pkl`（每个概念的潜在相关概念）交付给 B。

**说明**：上述图挖掘指标来自 A 的交付记录，本轮未重新运行 A 的生成脚本；这些脚本（`mine_concept_graph.py`、`community_detect.py`、`link_predict.py`、`build_community_meta.py`）不在 B/C 的交付包中，因此图本身无法从源头复现。

### 2.2 模块 B：图增强的 RAG

B 将问题解析、图定位、检索、语义筛选、排序和生成串成一条流水线。

**1）意图解析（规则，无 LLM）**
`intent_parser.py` 从自然语言中提取：
- `topic`（核心主题，中文主题用别名表映射为英文，如“板块构造 → plate tectonics”）；
- `year_range`（如“近五年 → (2022, 2026)”）；
- `paper_type`（综述、教程、预印本、荟萃分析）；
- `search_query`（更具体的限定检索式）；
- `is_overview`（概况类问题）与 `needs_topic_resolution`（是否需要 LLM 进一步解析）。

**2）主题整理（LLM）**
当规则无法确定主题，或问题为概况类且含中文时，调用 `resolve_search_topic` 或 `translate_topic`，将主题整理为英文检索词。

**3）图定位与关键词扩展**
`graph_api.GraphContext.locate(topic)` 将主题映射到图概念，匹配方式依次为：精确 → 去括号后精确 → 单复数 → 词级模糊（`contains` / `token`）。只有 `exact` 与 `singular` 被视为“可靠匹配”。

- 可靠匹配时：按档位获取社区 tier 与扩展概念；`novice`/`intermediate` 用共现近邻（`cooccur`），`expert` 用链路预测邻居（`link_pred`）。
- 不可靠匹配时：界面显示“图中没有可靠匹配 · 已转为 Acemap 主题检索”，退化为纯 Acemap 主题检索。这是正常的降级路径。

**4）检索（Acemap）**
`AcemapPilot.acemap.fetch()` 请求 `https://acemap.info/api/v1/work/search`，每词 50 条。检索词由「具体检索式 / 主题 / 图扩展词」去重组成。近期问题会附加 `order_by=publication_date`。

**5）语义筛选与向量检索**
`evidence_index.py` 使用 `sentence-transformers/all-MiniLM-L6-v2` 将论文标题与摘要片段向量化，存入 ChromaDB（`chroma_db/`），按余弦相似度排序。语义排序的前 `2 × top_k` 篇进入 LLM 相关性筛选，只保留能支持问题的论文。

**6）排序与个性化**
`rag_system._rank` 依次按「主题提及 → 类型匹配 → 兴趣亲和 → 年份/被引」排序。`interest_choices()` 提供 16 个社区作为可勾选兴趣（最多 3 个）。

**关于“个性化”的准确口径**：代码实现中，兴趣只影响同分次序（tie-break）、`expert` 档的扩展邻居（`expand`）和前 3 篇的 `recommended` 标记；它**不改变候选集合**，也**不进入结论提示词**。早期 README 中“影响论文入选”的说法已按实际代码修正。这一实现方式与项目目标中“个性化推荐”的期望存在差距，是本次评估重点关注的对象。

**7）结论生成（LLM）**
按档位选用 `prompts/novice.txt`、`prompts/intermediate.txt`、`prompts/expert.txt` 之一。三档的写作要求分别为：零基础讲解并附术语释义、对比方法与局限、直接给出判断并区分证据强度与推测。生成结果被解析为 2–4 句、带编号引用的结论；引用编号与论文卡片一一对应。

**8）结果结构**
`enhanced_retrieve()` 返回：`mode`、`conclusion`、`papers`（含 citation、url、年份、context、recommended）、`steps`（意图/社区/扩展/检索四个阶段的说明）、`warnings`、`generation_status`（`generated` / `insufficient_evidence` / `unavailable`）、`contexts` 以及 `retrieval` 诊断信息（检索词、年份、tier、放宽步骤、失败词、筛选状态等）。

### 2.3 三档配置

三档参数完全由界面下拉框决定，不做用户水平推断；`modes.resolve()` 为纯静态映射，非法值回退 `intermediate`。

| 参数 | novice | intermediate | expert |
|---|---|---|---|
| tier | basic | basic + subfield | subfield + niche |
| 年份范围 | 不限 | 不限 | 2021–2026 |
| 排序 | 被引 | 混合（新旧各 5） | 年份 |
| top_k | 8 | 10 | 12 |
| 扩展方式 | cooccur | cooccur | link_pred |
| 语言风格 | 入门讲解 + 白话释义 | 对比方法与局限 | 直接判断、区分证据强度 |

### 2.4 LLM 使用范围

全部生成式 LLM 调用集中在 `rag_generation._chat`，通过 DashScope 兼容接口，默认模型 `qwen3.8-flash`。四个调用点：

| 函数 | 用途 |
|---|---|
| `resolve_search_topic` | 提取英文主题与具体检索式 |
| `translate_topic` | 中文主题译为英文检索词 |
| `select_relevant_evidence` | 批量筛选相关论文 |
| `generate_conclusion` | 生成带引用的结论（格式失败时重试一次） |

非 LLM 组件：`intent_parser`（规则）、`graph_api`（查图）、`AcemapPilot`（拉取论文）、排序逻辑、`sentence-transformers`（嵌入模型，非生成式 LLM）。

**“离线”的准确含义**：本评估中的“离线回放”只离线化了 Acemap 的论文数据；上述四个 LLM 步骤以及 RAGAS 评估仍需联网与 DashScope 密钥。

---

## 三、运行架构

### 3.1 代码模块

| 文件 | 职责 |
|---|---|
| `gradio_app.py` | Web 界面：模式下拉框、兴趣勾选、问题输入、结果卡片与四步轨迹 |
| `rag_system.py` | 主流程 `enhanced_retrieve`，串联解析、定位、检索、筛选、排序、生成 |
| `intent_parser.py` | 规则式意图解析 |
| `modes.py` | 三档静态配置 |
| `graph_api.py` | 加载 pkl/csv，提供 `locate` / `match` / `expand` / `search_terms` / `glossary` |
| `rag_generation.py` | 四个 LLM 调用与结果校验 |
| `evidence_index.py` | MiniLM 嵌入 + ChromaDB 向量检索 |
| `AcemapPilot/acemap.py` | Acemap 请求与快照录制/回放 |
| `prompts/*.txt` | 三档提示词模板 |
| `eval/`（C 新增） | 测试问题、参考答案、录制/回放/评估脚本与结果 |

### 3.2 一次查询的数据流

```
用户问题 + 模式 + 兴趣
   ↓  intent_parser（规则）：topic / 年份 / 类型 / 概况
   ↓  [LLM] 主题整理（必要时）
   ↓  graph_api.locate → 可靠匹配？ ──否──→ 纯 Acemap 主题检索
   ↓                        └─是──→ 社区 tier + 扩展词（cooccur / link_pred）
   ↓  Acemap 检索（每词 50 条，串行限速）
   ↓  年份过滤 + _paper 结构化
   ↓  MiniLM 切分与向量化 → ChromaDB 语义排序
   ↓  [LLM] 相关性筛选（前 2×top_k 篇）
   ↓  _eligible（tier 过滤，证据不足时逐级放宽）
   ↓  _rank（主题提及 / 类型 / 兴趣 / 年份或被引）
   ↓  [LLM] 生成带引用结论
   ↓  结果：conclusion + papers + steps + warnings + generation_status
```

### 3.3 界面与运行方式

`gradio_app.py` 用 `gr.Blocks` 构建，前端通过自定义 JS 将模式、兴趣和问题打包进 `gr.EventData`，`_search()` 取参数后调用 `enhanced_retrieve()`，再渲染结论卡片、引用论文列表与四步轨迹。服务默认监听 `127.0.0.1:7860`。

运行依赖：`data_mining` 环境（Python 3.11.18），主要包版本与 `requirements.txt` 一致；`chroma_db/` 与 `.runtime/huggingface/` 为本地缓存目录，均已被 `.gitignore` 忽略。模型 `sentence-transformers/all-MiniLM-L6-v2` 已放置在 `.runtime/huggingface/hub/` 的快照目录中，可离线加载。

### 3.4 评估架构（模块 C）

- **问题集**：`eval/questions.csv`，15 条，三档各 5 条（由 C 起草，待组内确认）。
- **数据固定**：`eval/record.py` 用真实运行路径录制 Acemap 响应，生成 33 个快照；`eval/index.csv` 一行对应一个快照；回放时按（检索词、size、order_by）哈希定位。
- **参考答案**：`eval/references.json`，15 条（由 C 依据领域常识起草，未经组内核对）。
- **评估脚本**：`eval/export_ragas_inputs.py` 导出 question/answer/contexts；`eval/run_ragas.py` 用 ragas 0.4.4 评分，评估 LLM 为 `qwen3.8-max-0902`，嵌入为本地 MiniLM；`eval/score_one_metric.py` 逐指标评分并即时保存。

---

## 四、已完成的工作与验证

### 4.1 代码修复（模块 C 对 B 代码的改动，均在 C 仓库中）

| 改动 | 内容 |
|---|---|
| 失败可见性 | `AcemapPilot/acemap.py` 新增 `SnapshotMissing`；`_collect` 记录每个失败词的原因（如 HTTP 403、连接中断、离线快照缺失）并写入提示 |
| 错误包装 | 向量模型加载失败转为可读的 `SearchError` |
| LLM 稳定性 | `temperature` 默认改为 0（可环境变量覆盖）；筛选 `max_tokens` 128→256（截断重试 512）；结论 650→900；主题解析 120→240；翻译 80→160 |
| 文档修正 | `README_B.md` 修正兴趣作用描述，补充 UA 与快照回放约定 |

### 4.2 环境与工具

- `ragas 0.4.4` 已安装；`huggingface-hub` 由 2.4.0 降级至 0.36.2（与 transformers 4.57.6 兼容）；`datasets` 由 5.2.0 降级至 4.8.5（满足 ragas ≥4.0.0）。
- MiniLM 离线加载验证通过，输出维度 `(1, 384)`。
- Acemap 的 403 根因已定位：`python-requests` 默认 UA 被拒；项目自定义 UA 正常。已排除频率与关键词因素。
- 残留的不完整 `handover_B_to_C/GraphRAG4Scholar_B/venv` 已删除。

### 4.3 错误路径验证（真实或模拟触发）

| 路径 | 结果 |
|---|---|
| 截断后重试（模拟 `finish_reason=length`） | 通过：max_tokens 依次为 256、512 并得到结果 |
| 双重截断 | 通过：抛出 `TruncatedOutput`（`GenerationError` 子类），由上层降级 |
| 快照部分缺失 / 全部缺失 | 通过：缺失词记为原因，其余正常；全部缺失返回可读 `SearchError` |
| 向量模型加载失败 | 通过：转为“向量模型未就绪”提示 |
| temperature 默认与覆盖 | 通过：默认 0.0，环境变量可覆盖 |

### 4.4 回放与可复现性

15 条问题全部可离线回放。修改前两轮回放中 N5、I3、E2 等结果不同；修改后（`temperature=0` + 提高 max_tokens），13/15 两轮完全一致，仍有 N3（结论引文校验失败）与 E4（论文数 4↔3）不同。根因是 LLM 输出仍有残余非确定性，已按决定不做响应缓存。

### 4.5 RAGAS 评分结果（单次估计）

评估 14 题（E3 按决定排除）：

| 指标 | n | 均值 | 最小 | 最大 |
|---|---|---|---|---|
| Faithfulness | 14 | 0.783 | 0.400 | 1.000 |
| AnswerRelevancy | 14 | 0.570 | 0.145 | 0.856 |
| ContextRecall（参考答案待核对） | 14 | 0.482 | 0.000 | 1.000 |

注：原额度耗尽后已按方案 A 切换到生成模型 `deepseek-v4.1-flash`、评估模型 `kimi-k3.1-pro`，并从头重跑 15 题导出与 14 题评分；上表为新模型结果。切换前后的分数不可直接比较。

**解读限制**：
- 分数为单次估计，评估 LLM 本身有波动（N4 两次 faithfulness 分别为 0.923 与 0.714）；不适合题目间显著性比较。
- ContextRecall 依赖由 C 起草、未经组内核对的参考答案。
- 评估 LLM（kimi-k3.1-pro）与生成 LLM（deepseek-v4.1-flash）通过同一兼容接口访问，仍存在同平台偏差；两者的波动未系统测量。

---

## 五、已知问题与局限

### 5.1 Faithfulness 偏低

I1、E1 等题的 Faithfulness 较低（最低 0.2）。对 I1 的关键词核对显示：结论中多数细节有原文支撑，但 `Cocos`、`tear`、`Qilian`、过渡带等表述在上下文中找不到原文，属于生成时的补充推断。是否为真实幻觉需人工逐条核对，尚未完成。

### 5.2 个性化效果受限

- 兴趣不改变候选集合，只影响同分排序、扩展邻居与推荐标记；
- 多数主题在图中有“可靠匹配”的比例有限（如 `RAG`、`板块构造` 中文形式、`large language model` 均未精确命中），这些题看不到个性化差异；
- `subduction zone` 被 `token` 方式误匹配到 `benthic zone`（不可靠，未用于扩展）。

### 5.3 E3 的情况（仅记录，不作评判依据）

`subduction zone 近五年有哪些综述论文` 返回 0 篇、`insufficient_evidence`。观察：送筛的 24 篇候选标题均无综述关键词；类型偏好只作为排序加分，未参与召回；Acemap 关键词检索不能可靠召回综述。按组内决定，不再深究，E3 不计入任何指标解读、结论或评判。

### 5.4 结果非严格确定

`temperature=0` 后仍有两题两轮结果不同。评估时应报告多次运行的分布，而非单次结果。本轮尚未系统测量波动范围。

### 5.5 外部服务与数据

- Acemap 连接不稳定：实时访问中多次出现 10054 连接重置；默认 UA 返回 403。
- 主题翻译与结论依赖 DashScope 密钥与网络；无密钥时相关步骤报错。
- 图无法从源头复现（A 的生成脚本未交付）。

### 5.6 工程状态

- 本报告涉及的改动尚未提交到 git；`eval/` 下的中间产物是否纳入版本控制未定。
- `requirements.txt` 未包含 ragas，`huggingface-hub` 与 `datasets` 版本为手动调整，未形成锁定文件。
- 修改后的界面未在浏览器中做完整端到端测试。
- 环境与依赖的 pip 报告存在 datasets/huggingface-hub 版本冲突提示（本项目不直接使用 datasets）。

### 5.7 尚未完成的模块 C 任务

- 召回提升（≥10% 阈值）对比：判定方式（人工标注或 LLM 判相关）未定，baseline 对比未运行。
- 三档与个性化的系统化对比实验未执行。
- 图挖掘评估复核、A0 海报、1–2 分钟演示视频未开始。

---

## 五之二、结构性对比（额度耗尽后的补充测试）

DashScope 免费额度在后续测试中耗尽（HTTP 403，`insufficient_quota`），需要 LLM 的步骤暂不可用。为此做了“固定筛选为全部通过、固定结论为占位”的结构性对比，只验证检索与排序层：

- **三档对比**：三个问题在 novice / intermediate / expert 下的论文集合均不同，档位配置确实生效。
- **个性化**：expert / intermediate 下有/无兴趣的最终集合与顺序不同；novice 相同。注意这反映的是扩展词与排序的差异，进入筛选前的候选集合仍相同。
- **基线对比**：增强版与纯 Acemap 的候选集合明显不同（增强版独有 5–9 篇）。但这只是候选召回差异，**不是**相关性召回提升的验证，不能据此判断 10% 阈值。
- **结构性确定性**：同一问题三次独立运行，首篇论文 ID 一致；此前的波动来自 LLM 步骤。

需要 LLM 参与的三档答案质量对比、个性化答案对比、RAGAS 重评，均需等待额度恢复。结果文件：`eval/compare_structure_*.json`；说明见 `eval/status_report.md`。

## 五之三、召回提升阈值验证

用同一参考答案、同一确定性评估配置（`kimi-k3.1-pro`，`temperature=0`），对 14 题分别计算增强版与基线的 ContextRecall：

- 均值：基线 0.143，增强 0.400，**绝对提升 25.7 个百分点**，数值上超过 10% 阈值。
- 但只有 5/14 题出现正提升，1 题下降，其余持平；相对提升口径因多数基线为 0 而无法计算。
- 参考答案为概括性长句、与具体摘要粒度不匹配，导致大量 0 分，因此该结论不稳健。需组内核对或改写参考答案后重评。

评估模型波动的根因已确认：`kimi-k3.1-pro` 只接受 `temperature=0`；此前未显式指定导致服务端使用非零默认值。修正后三次重复评分完全一致，故未采用重复评分取均值的方案。不建议更换测试题目——波动与题目无关，换题会使快照与全部评估结果作废。

## 六、结论与建议

系统已具备完整功能链路：从自然语言问题到带引用的结论，支持三档模式、兴趣勾选与图增强检索；评估用的数据固定、回放、评分工具链已经建立，并完成了 14 题的三指标试评估。

当前最需要处理的三件事是：

1. **核对参考答案**（`eval/references.json`）并人工抽检 Faithfulness 低分的结论，再据以修订报告中的评分结论；
2. **确定召回判定方式**，补做增强版与纯 Acemap baseline 的对比，验证是否达到 10% 提升阈值；
3. **补做三档与个性化对比**，完成海报与演示视频。

以上均依赖组内对评估口径的决定，建议在下一轮对齐会上明确后执行。

---

## 附录：主要文件清单

- 框架与运行：`gradio_app.py`、`rag_system.py`、`intent_parser.py`、`modes.py`、`graph_api.py`、`rag_generation.py`、`evidence_index.py`、`AcemapPilot/acemap.py`、`prompts/*.txt`
- A 的数据：`concept_community.pkl`、`link_pred_result.pkl`、`community_meta.csv`、`full_concept_edges.csv`、`glossary.csv`
- C 的评估：`eval/questions.csv`、`eval/references.json`、`eval/record.py`、`eval/index.csv`、`eval/export_ragas_inputs.py`、`eval/run_ragas.py`、`eval/score_one_metric.py`、`eval/retry_failed_ragas.py`
- C 的结果与记录：`eval/ragas_inputs.jsonl`、`eval/ragas_scores.json`、`eval/replay_*.json`、`eval/status_report.md`
