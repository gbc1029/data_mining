# GraphRAG4Scholar 模块 B

应用从自然语言问题出发，按需从 Acemap 取论文，用 A 的概念图和三档配置检索，再根据实际摘要片段生成带编号的结论。

## 解压后启动

将交接包完整解压到 E 盘，在包含 `gradio_app.py` 的目录打开 PowerShell。包内带有前端资源、Acemap 请求客户端和运行所需的 A 图数据与三档配置。下面的安装组合已在 Windows x64 与 CPython 3.11 上验证。首次安装需要 `uv`，以及能访问 Python 包源和模型下载服务的网络。

```powershell
$runtime = Join-Path $PWD '.runtime'
$env:TEMP = Join-Path $runtime 'tmp'
$env:TMP = $env:TEMP
$env:UV_CACHE_DIR = Join-Path $runtime 'uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $runtime 'python'
$env:HF_HOME = Join-Path $runtime 'huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:GRADIO_TEMP_DIR = Join-Path $runtime 'gradio'
$env:XDG_CACHE_HOME = Join-Path $runtime 'cache'
$env:TORCH_HOME = Join-Path $runtime 'torch'
New-Item -ItemType Directory -Force -Path $env:TEMP, $env:GRADIO_TEMP_DIR | Out-Null

uv venv --managed-python --python 3.11 venv
uv pip install --python .\venv\Scripts\python.exe `
    --index https://download.pytorch.org/whl/cpu `
    -r requirements.txt -c constraints-win-py311.txt "torch==2.9.1+cpu"
```

在同一个窗口配置自己的百炼密钥，然后启动。已经安装依赖后，再次运行时从缓存设置和启动命令开始即可。

```powershell
if (-not $env:DASHSCOPE_API_KEY) {
    $key = Read-Host '输入自己的百炼 API Key' -AsSecureString
    $env:DASHSCOPE_API_KEY = [System.Net.NetworkCredential]::new('', $key).Password
}
.\venv\Scripts\python.exe gradio_app.py
```

用浏览器打开终端打印的 `Running on local URL`，保持 PowerShell 窗口运行。试用时可点击「板块构造近五年」并搜索。首次查询会下载 `sentence-transformers/all-MiniLM-L6-v2`，摘要片段及向量保存在 `chroma_db/`。实际查询需要能访问 Acemap 和百炼服务。

默认生成模型是 DashScope 的 `qwen3.8-flash`，从进程环境读 `DASHSCOPE_API_KEY`。可通过 `GRAPHRAG_LLM_BASE_URL`、`GRAPHRAG_LLM_MODEL`、`GRAPHRAG_LLM_API_KEY` 指定兼容服务。兼容地址填到 `/v1`，程序会自行拼接 `/chat/completions`。已完成相关性筛选的论文与证据片段在结论生成失败时仍会显示。

## 给模块 C 的评估入口

```python
from rag_system import enhanced_retrieve

question = "近五年板块构造有哪些研究进展？"
result = enhanced_retrieve(question, mode="expert", interests=[12])
baseline = enhanced_retrieve(question, mode="expert", baseline=True)
```

- `conclusion` 是答案，`contexts` 是实际传给生成模型的摘要片段。`papers` 包含顺序一致的引文编号、Acemap ID、链接、年份、上下文和推荐标记。
- `generation_status` 区分已生成、证据不足和服务不可用。`retrieval` 记录检索词、年份、兴趣、tier、实际放宽步骤和失败的 Acemap 查询。
- 普通查询先按语义选取最多 `2 × top_k` 篇，批量核对其标题和实际摘要片段是否能支持原问题，再按模式和兴趣排序。`retrieval.screened` 与 `screening_status` 记录筛选尝试和状态。筛选服务失败时只显示状态提示。
- 简称和宽泛概况问题需要整理检索词时，使用一次短模型调用提取完整学术主题。概况题命中图中主题时按三档规则扩展，图外最多向 Acemap 发送两个检索词。回答概括证据覆盖的方面，明确的时间、比较对象和应用场景继续约束检索与作答。
- `baseline=True` 使用单一检索词并保留 Acemap 顺序。问题的时间条件、模式年份限制、`top_k` 上限、摘要切分和生成设置与增强版本一致。与纯 Acemap 的对比衡量整条增强流程，包括相关性筛选。兴趣对照使用同一档位和相同候选来源，改变 `interests`。
- 传入 `paper_source(keyword, size, *, order_by=None)` 可录制或重放同批 Acemap 响应。评估时按检索词、条数和排序参数保存原始候选，记录生成模型版本，将同一份响应重放给各版本。先导项目的快照以 20 或 30 篇为主，本版每词请求 50 篇，正式对照宜用同一次采集的数据。

兴趣列表读取 A 提供的 `community_meta.csv`。兴趣只影响排序中的同分次序、扩展（link）邻居与推荐标记，不改变候选集合，也不影响结论提示词；引用列表保持当前档位规定的顺序，最多标记三篇与兴趣匹配的论文。

访问约定：Acemap 会拒绝 `python-requests` 默认 User-Agent（返回 403），所有请求必须经 `AcemapPilot.acemap.fetch()`，它使用项目 UA `AI3602-PilotProject/1.0`。离线回放时快照按（检索词、size、order_by）哈希定位，缺失快照会被记为失败原因“离线快照缺失”，并写入 `warnings`，不会中断整个查询。

Acemap 的论文类型字段在现有响应中不统一，综述、教程等需求以标题和链接偏好参与入选。近期窗口使用 Acemap 的发表日期排序召回，并在本地严格筛年份。中间档按最近五个自然年划分新旧论文，两组各取五篇，缺额由其余候选补齐。社区标签取 Acemap 概念列表前六项中、通过图精确或单复数匹配的实体。只由图扩展召回的论文需有原主题的概念或文本证据。证据不足时按 A 的顺序放宽社区 tier，显式年份始终生效。
