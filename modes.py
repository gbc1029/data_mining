"""
三档用户模式的静态配置。

约定（与《期末项目协作指南》4.2 / 4.3 一致）：
    - 档位只来自界面下拉框，这里不做任何用户水平推断；
    - resolve(mode) 纯静态映射，同一档对任何 topic 返回完全相同的值；
    - 返回值固定 9 个键，B 侧不许硬编码任何阈值。

用法：
    from modes import resolve
    cfg = resolve("expert")
"""

MODES = ("novice", "intermediate", "expert")

PROFILES = {
    "novice": {
        "tier": ["basic"],
        "year_range": None,
        "sort": "cited",
        "quota": None,
        "expand": "cooccur",
        "top_k": 8,
        "profile": "你是给零基础本科生讲解的研究助理。",
        "prompt_file": "prompts/novice.txt",
    },
    "intermediate": {
        "tier": ["basic", "subfield"],
        "year_range": None,
        "sort": "mixed",
        "quota": {"old": 5, "new": 5},
        "expand": "cooccur",
        "top_k": 10,
        "profile": "你是给高年级本科生和研一学生讲解的研究助理。",
        "prompt_file": "prompts/intermediate.txt",
    },
    "expert": {
        "tier": ["subfield", "niche"],
        "year_range": (2021, 2026),
        "sort": "year",
        "quota": None,
        "expand": "link_pred",
        "top_k": 12,
        "profile": "你是与领域专家讨论的研究助理。",
        "prompt_file": "prompts/expert.txt",
    },
}

DEFAULT_MODE = "intermediate"


def resolve(mode):
    """按用户选择的档位返回参数；非法值回退 intermediate，不抛异常。"""
    if mode not in PROFILES:
        mode = DEFAULT_MODE
    cfg = dict(PROFILES[mode])          # 复制，避免调用方改坏全局配置
    cfg["mode"] = mode
    return cfg


if __name__ == "__main__":
    for m in MODES:
        cfg = resolve(m)
        print(f"[{m}] tier={cfg['tier']} year={cfg['year_range']} "
              f"sort={cfg['sort']} top_k={cfg['top_k']}")
    print(f"[回退] resolve('Expert') -> {resolve('Expert')['mode']}")
    print(f"[回退] resolve(None)     -> {resolve(None)['mode']}")
