"""文献一句话概要（gist_cn）：写作端与评审端共用的第二份真实素材。

背景
----
`01_literature.json` 的 `gist_cn` 是检索阶段落盘的一句话概要。它一直被
`_role_citation_block` 确定性拼进正文引用块，却**从未进入写作提示的事实包**：
模型写「研究现状」时只看得到题名，于是凭题名脑补文献结论。实测里模型把
「测试八周体育干预对农村中学儿童心血管风险的影响」扩写成「发现干预方案与
学校既有日程的匹配程度直接决定了实施质量，回应了 SERVQUAL 的响应性与移情性」，
评审官判为编造——它编得有鼻子有眼，但依据根本不存在。

上一轮把 gist 补进了评审事实基础（`_gist_gt`），这一轮补进写作事实包，且
**两端共用同一个 ``gist_items()`` 渲染**，条目逐字同源。只放宽一边是上轮
已经踩过的坑。

另外修一个确定性缺陷：`gist_cn` 为「未知」「无」「N/A」等占位值时，旧代码
会把它当成真实概要输出成「：未知」，既难看又诱导模型替它编内容。
"""
from __future__ import annotations

import re
from typing import Any

_PLACEHOLDERS = {"", "未知", "无", "暂无", "n/a", "na", "none", "null",
                "unknown", "not available", "未标注", "-", "--", "?"}

_WRITER_HEADER = """## 文献一句话概要（检索阶段落盘的真实素材，[编号] 与参考文献清单一致）

以下每行是对应文献的一句话概要，也是正文引用块的来源。**只有列出的文献有概要**；
没有列出的文献你只有题名，只能定性推断题名，不得补造结论。

使用边界（违反即判编造）：
1. 可据此转述该文献的研究对象、采用的方法、使用的数据与主要结论方向；
2. **概要里没写的结论、机制、因果链条一律不得推断补写**。实测踩过坑：把
   「测试八周体育干预」扩写成「发现匹配程度决定实施质量、回应了某理论维度」，
   这属于编造文献结论，不是合理转述；
3. 概要中出现的数字属于已提供的真实素材，可原样引用，但须写明「该文献报告……」
   「[N] 该研究……」等归属，不得写成「本文实验得到」；
4. 不得把 [N] 的概要内容安到 [M] 或本文头上；题名与概要冲突时以概要为准。
"""

_JUDGE_HEADER = """【文献一句话概要：检索阶段落盘的元数据，正文引用块会原样采用】

以下每条与参考文献编号一一对应。其中出现的样本量、记录数等数字属于
**已提供的真实素材**，不得据此判为「事实清单之外」或编造；但若正文把这些
数字安到别的文献编号上，仍属张冠李戴。概要没写而正文写了的结论，属于编造。
"""


def normalize_gist(value: Any) -> str:
    """清洗一句话概要：占位值一律视为缺失（旧版会把「未知」写进正文）。"""
    if not isinstance(value, str):
        return ""
    s = " ".join(value.split()).strip()
    if s.strip().lower() in _PLACEHOLDERS:
        return ""
    return s


def gist_items(papers: list[dict] | None, max_items: int = 80,
               max_chars: int = 9000) -> list[str]:
    """渲染带编号的概要条目。编号与 build_references 的连续编号一致，不重排。"""
    out: list[str] = []
    used = 0
    for i, p in enumerate(papers or [], 1):
        if not isinstance(p, dict):
            continue
        g = normalize_gist(p.get("gist_cn"))
        if not g:
            continue
        line = f"[{i}] {g}"
        if len(out) >= max_items or used + len(line) > max_chars:
            break
        out.append(line)
        used += len(line) + 1
    return out


def gist_block(papers: list[dict] | None, audience: str = "writer") -> tuple[str, dict]:
    """返回 (可直接拼进提示词的块, 统计)。写作端与评审端的**条目逐字相同**。"""
    items = gist_items(papers)
    if not items:
        return "", {"n_total": len(papers or []), "n_used": 0}
    head = _JUDGE_HEADER if audience == "judge" else _WRITER_HEADER
    return (head + "\n".join(items) + "\n",
            {"n_total": len(papers or []), "n_used": len(items)})


# ── 文献池与检索式的主题一致性（只做提示，不改闸门） ───────────────────
# 用「词的前 5 个字符」代替词干化：predict / prediction / predicting 都归到 predi，
# 不需要维护词干表，也不会因为单复数误判。
_STOP = {
    "with", "from", "using", "based", "study", "studies", "research", "approach",
    "novel", "review", "analysis", "model", "models", "data", "for", "and", "the",
    "of", "in", "on", "to", "systematic", "comparison", "comparative", "framework",
    "method", "methods", "application", "applications", "toward", "towards", "via",
}


def _keys(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{4,}", str(text).lower())
            if w not in _STOP}


def topic_overlap(query: str, papers: list[dict] | None) -> float:
    """检索式的主题词至少命中一篇文献题名的篇数占比（0–1）。

    这是粗粒度的确定性提示，只用于「文献池可能整体跑题」的预警，
    **不用于过滤文献、不参与任何闸门**——换编号会打乱正文 [N] 引用。
    """
    papers = [p for p in (papers or []) if isinstance(p, dict)]
    if not papers or not query:
        return 0.0
    terms = _keys(query)
    if not terms:
        return 0.0
    hit = sum(1 for p in papers[:80] if _keys(p.get("title", "")) & terms)
    return round(hit / min(len(papers), 80), 3)
