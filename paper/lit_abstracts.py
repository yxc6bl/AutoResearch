"""文献摘要素材：把 01_literature.json 里**真实存在**的摘要接进零编造事实包。

为什么需要
----------
检索 Agent 抓到的文献里，有相当一部分带 Crossref/OpenAlex 返回的 abstract。
但旧版写作规格写死「你只有题名/作者/来源/年份，没有摘要」，评审官的事实基础
也写死「清单不含摘要」——两边一致地把真实素材锁在门外。结果是综述、开题、
案例研究只能靠题名做定性推断，正文空转、评审分长期卡在 60–80。

本模块是**单一事实源**：写作端的 fact_pack 与评审端的 G04 事实基础调用同一个
``abstract_block()``，保证「模型看到什么」和「评审官核对什么」逐字一致。
一边放宽、另一边不放宽是最危险的：会把真实转述误判成编造，或把编造误当真实。

零编造边界
----------
摘要是【别人论文】的内容，不是本文的实验结果。实测还发现，只要把带具体数字的
摘要交给模型，它就会把「可以引用 [8] 摘要里的数字」泛化成「可以给 [9] 也补一个
数字」（实测编造出「Kaggle 918 条记录」「7 万条记录」，两条都被 G04 抓住）。
因此本模块默认对摘要做**数值脱敏**：保留研究问题、方法路线、数据来源、结论方向
这些定性信息，把所有具体数值替换成「〔数值略〕」。写作端与评审端都看不到数字，
正文一旦出现具体数字，就能被事实基础确定性判定为编造。
"""
from __future__ import annotations

import os
import re
from typing import Any

# 摘要接入默认关闭。四轮实测（见 docs/CAPABILITY.md 10.8）都未能超过基线：
# 一旦把摘要交给模型，它就会给没有依据的文献补数字；脱敏后又会与正文引用块里
# 既有的 gist 数字冲突，评审官判定失稳。能力保留，用环境变量显式开启做实验。
_ENV_FLAG = "AR_LIT_ABSTRACTS"


def enabled() -> bool:
    """是否启用摘要接入。默认关闭，保持已验证的零幻觉基线行为。"""
    return os.environ.get(_ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")

# 占位符 / 版权声明：这些不是摘要，放进去只会给模型制造噪声
_PLACEHOLDER = re.compile(
    r"^\s*(no\s+abstract|abstract\s+not\s+available|not\s+available|n/?a|"
    r"none| unavailable|暂无摘要|无摘要)\b", re.I)
# JATS / HTML 标签（Crossref 的摘要常带 <jats:p>、<jats:title>）
_TAG = re.compile(r"<[^>]{1,80}>")
_WS = re.compile(r"\s+")
# 数值脱敏：不碰紧跟字母的写法（1D-CNN、3D、2型糖尿病），只替换独立的数字串
_NUM = re.compile(r"(?<![\w])\d[\d,.]*%?(?![\w])")

_HEADER = """## 文献摘要节选（真实数据，可据此转述；[编号] 与参考文献清单一一对应）

以下每行的 [编号] 与下方参考文献清单严格一致。**只有列出的文献有摘要**；
没有列出的文献，你仍然只能依据题名定性推断，不得补造摘要、结论或数字。

**注意：摘要中的具体数值已统一替换为「〔数值略〕」。** 你看不到任何样本量、
准确率、年份之外的数字，因此正文也**严禁出现任何具体数字**（样本量、记录数、
准确率、AUC、F1 等）。需要说明规模时，写「较大规模队列」「多中心样本」这类定性表述。

使用边界（违反即判编造）：
1. 可转述该文献的研究问题、方法路线、数据来源与结论方向；这些内容必须能在
   本节对应编号的摘要节选中找到依据；
2. 凡引用摘要内容，用「该文献报告……」「[N] 该研究……」「据 [N] 的摘要，……」
   这类带归属的句式，不得写成无条件事实；
3. 不得把某篇文献的方法、数据集或结论安到另一篇文献或本文头上；
4. 摘要是原文节选，不是全文；不要推断摘要之外的实验设置、超参数或贡献；
   摘要只写「比较了 A 与 B」时，不得扩写成「发现 A 优于 B」。
"""


def clean_abstract(text: Any, limit: int = 420,
                   strip_numbers: bool = True) -> str:
    """清洗单篇摘要：去标签、压空白、去占位符、**数值脱敏**、截断。不可用返回空串。"""
    if not isinstance(text, str):
        return ""
    s = _TAG.sub(" ", text)
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    s = _WS.sub(" ", s).strip()
    if len(s) < 40 or _PLACEHOLDER.match(s):
        return ""
    if strip_numbers:
        s = _NUM.sub("〔数值略〕", s)
    if len(s) > limit:
        # 尽量在句末截断，避免把半句话塞给模型
        cut = s[:limit]
        dot = max(cut.rfind(". "), cut.rfind("。"))
        s = (cut[:dot + 1] if dot >= limit * 0.6 else cut.rstrip()) + "…"
    return s


def _quality(p: dict) -> float:
    """排序用：优先贴合度高、被引多的文献（只影响「选哪几篇」，不改变编号）。"""
    out = 0.0
    for key in ("fit_score", "relevance_score", "quality_score", "select_score"):
        v = p.get(key)
        if isinstance(v, (int, float)):
            out += float(v)
    c = p.get("cited_by_count")
    if isinstance(c, (int, float)):
        out += min(float(c), 200) / 1000.0
    return out


def abstract_block(papers: list[dict] | None,
                   per_paper: int = 300,
                   max_papers: int = 36,
                   max_chars: int = 9000) -> tuple[str, dict]:
    """返回 (可直接拼进提示词的摘要块, 统计)。

    ``papers`` 必须与 01_literature.json 的 papers 顺序一致：``build_references``
    用 enumerate 从 1 连续编号，所以这里第 i 篇就是 [i]，不能重排后重新编号。
    筛选按质量排序，但**输出按编号升序**，避免正文引用编号对不上。

    没有任何可用摘要时返回 ("", stats)，调用方走原来的「仅题名」路径，行为不变。
    """
    stats = {"n_total": 0, "n_with_abstract": 0, "n_used": 0, "chars": 0}
    papers = [p for p in (papers or []) if isinstance(p, dict)]
    stats["n_total"] = len(papers)
    cand: list[tuple[int, dict, str]] = []
    for i, p in enumerate(papers, 1):
        ab = clean_abstract(p.get("abstract"), per_paper)
        if ab:
            cand.append((i, p, ab))
    stats["n_with_abstract"] = len(cand)
    if not cand:
        return "", stats

    cand.sort(key=lambda t: -_quality(t[1]))
    picked: list[tuple[int, str]] = []
    used = 0
    for i, p, ab in cand:
        title = str(p.get("title") or "").strip()[:110]
        year = str(p.get("year") or "").strip()
        line = f"[{i}] 《{title}》" + (f"（{year}）" if year else "") + f"：{ab}"
        if len(picked) >= max_papers or used + len(line) > max_chars:
            break
        picked.append((i, line))
        used += len(line) + 1
    if not picked:
        return "", stats
    picked.sort(key=lambda t: t[0])
    body = "\n".join(line for _, line in picked)
    block = _HEADER + "\n" + body + "\n"
    stats["n_used"] = len(picked)
    stats["chars"] = len(block)
    return block, stats


def load_papers(base_dir) -> list[dict]:
    """从产物目录读 01_literature.json 的 papers（读不到返回空列表）。"""
    from pathlib import Path
    p = Path(base_dir) / "01_literature.json"
    if not p.exists():
        return []
    try:
        import json
        d = json.loads(p.read_text(encoding="utf-8"))
        return (d.get("papers") or []) if isinstance(d, dict) else (d if isinstance(d, list) else [])
    except Exception:
        return []
