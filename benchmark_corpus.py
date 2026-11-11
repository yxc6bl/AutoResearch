#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""范文对标模块（AutoResearch 升级件）。

直接回应一个问题：「我们生成的论文，有没有跟高质量范文做过内容质量比对？」

本模块从 corpus/ 读取真实高质量本科毕设范文（如 北大2024、清华2008），
抽取一组「质量指纹」（方法原创性 / 章节数 / 文献数 / 顶会 venue / 公式数 /
消融·跨数据集信号 / 严谨度信号），再对一篇生成论文抽取同一组指纹，
产出可机器比较的「差距报告」（JSON + 可读摘要）。

用法：
    python benchmark_corpus.py <paper.md> [--corpus DIR] [--out report.json]

无网络依赖，纯正则指纹；可作为 EvalAgent 的一个信息性参考维度（corpus_benchmark）。
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

# ── 顶会 / 顶刊 venue 信号（用于识别范文是否站在方法前沿） ────────────────
TOP_TIER_VENUES = [
    "CVPR", "ICCV", "ECCV", "NeurIPS", "NIPS", "ICML", "ICLR", "AAAI",
    "SIGGRAPH", "IJCAI", "ACL", "EMNLP", "KDD", "WWW", "SIGIR",
    "TPAMI", "IEEE Transactions on Pattern Analysis",
    "Nature", "Science", "Cell", "IEEE Access",
]

# 方法原创性信号：明确提出新框架 / 算法 / 模型 / 方法 / 网络 / 模块 …
_NOVELTY_RE = re.compile(
    r'(本文提出|本文设计|本文构建|本文首次|本文给出|提出一种|提出了一种|设计了一种|构建了一种)'
    r'[^\n。]{0,40}?(框架|算法|模型|方法|网络|模块|策略|流程|机制|结构|系统|范式)'
)
_INNOV_SEC_RE = re.compile(r'创新点|本文贡献|独立见解|本文的创新点|主要贡献')
# 范文多为纯文本（PDF 抽取），章节用「第一章 引言」而非 Markdown 标题，故放宽匹配；
# 但须限定为「章标题行」（行首、可带 Markdown # 与空格），避免把正文里「见第3章」也算进去。
_CHAPTER_RE = re.compile(r'(?:^|\n)\s*#{0,4}\s*第([0-9一二三四五六七八九十百]+)章')
_REF_ITEM_RE = re.compile(r'^\s*\[\d+\]', re.M)
_FORMULA_RE = re.compile(r'式\s*[（(]\s*\d|公式|\\\(|\\\[')
_ABLATION_RE = re.compile(r'消融|超参数.*搜索|网格搜索|参数寻优|调优实验|敏感性分析')
_MULTI_DS_RE = re.compile(r'跨数据集|多个数据集|多数据集|在不同.*数据集')
_DATA_SCALE_RE = re.compile(
    r'(\d{3,})\s*(?:个|条|例|名|位|张)?\s*(?:样本|样本数|数据|病例|受试者|患者|个体|记录|实例)')
_SIG_RE = re.compile(r'显著性|t\s*检验|p\s*[=＜<]|p值|置信区间')
_CV_RE = re.compile(r'交叉验证|cross.validation|\d+\s*折')
_SEED_RE = re.compile(r'随机种子|多种子|种子\b')


def extract_fingerprint(text: str) -> dict:
    """抽取一篇论文/范文的质量指纹（全部为可机读信号）。"""
    novelty = _NOVELTY_RE.findall(text)
    chapters = set(_CHAPTER_RE.findall(text))
    # 参考文献块：范文与生成论文的「参考文献」可能多次出现（图注/正文提及），
    # 取其后含 [N] 条目最多者作为真正的参考文献表。
    _heads = list(re.finditer(r'参考文献', text))
    ref_block = ""
    if _heads:
        best = max(_heads, key=lambda mm: len(_REF_ITEM_RE.findall(text[mm.start():])))
        ref_block = text[best.start():]
    ref_items = _REF_ITEM_RE.findall(ref_block)
    venues = sorted({v for v in TOP_TIER_VENUES if re.search(re.escape(v), text, re.I)})
    scales = [int(x) for x in _DATA_SCALE_RE.findall(text)]
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    return {
        "cjk_chars": cjk,
        "novelty_mentions": len(novelty),
        "proposes_method": bool(novelty),
        "has_innovation_section": bool(_INNOV_SEC_RE.search(text)),
        "chapter_count": len(chapters),
        "ref_count": len(ref_items),
        "top_tier_venues": venues,
        "top_tier_count": len(venues),
        "formula_signals": len(_FORMULA_RE.findall(text)),
        "ablation_tuning": bool(_ABLATION_RE.search(text)),
        "multi_dataset": bool(_MULTI_DS_RE.search(text)),
        "dataset_scale_max": max(scales) if scales else None,
        "has_significance": bool(_SIG_RE.search(text)),
        "has_cv": bool(_CV_RE.search(text)),
        "has_multiseed": bool(_SEED_RE.search(text)),
    }


def _corpus_files(corpus_dir: str):
    """只收能读的中文范文。PDF 提取失败的 txt（cid 乱码/纯英文扫描噪声）
    汉字占比极低，混进参照库会把标尺拉歪，必须挡在外面。"""
    files = []
    for p in sorted(glob.glob(str(Path(corpus_dir) / "*.txt"))):
        name = Path(p).name.lower()
        if name.startswith("readme") or "schema" in name:
            continue
        text = Path(p).read_text(encoding="utf-8", errors="ignore")
        han = sum(1 for c in text if "一" <= c <= "鿿")
        if han < 1000 or (text and han / max(1, len(text)) < 0.05):
            continue
        files.append(p)
    return files


def reference_profile(corpus_dir: str = "corpus") -> dict:
    """从高质量范文里抽出「专家评阅标尺」，供评审 Agent 当作参照注入。

    这里不是训练模型，而是把清北范文的可观测质量特征整理成可核对的标尺：
    篇幅、章节、文献、原创性、验证强度、论证密度。评阅时用它判断
    「这篇论文写得像不像一篇好论文」，而不是只检查模板字段齐不齐。
    """
    docs = []
    for cf in _corpus_files(corpus_dir):
        t = Path(cf).read_text(encoding="utf-8", errors="ignore")
        fp = extract_fingerprint(t)
        cjk = max(1, fp.get("cjk_chars") or 1)
        # 论证密度：因果/转折/证据类连接词每千字出现次数
        causal = len(re.findall(r"因此|由于|从而|进而|由此|然而|相比之下|实验表明|结果表明|说明|证明", t))
        fp["argument_markers_per_1k"] = round(causal * 1000 / cjk, 2)
        fp["figure_table_refs"] = len(re.findall(r"图\s*\d|表\s*\d|式\s*\(|公式\s*\d", t))
        docs.append({"file": Path(cf).name, "fingerprint": fp})

    def _agg(key, agg=max):
        vals = [d["fingerprint"].get(key) for d in docs if d["fingerprint"].get(key) is not None]
        return agg(vals) if vals else None

    typical = {
        "cjk_chars": _agg("cjk_chars"),
        "chapter_count": _agg("chapter_count"),
        "ref_count": _agg("ref_count"),
        "novelty_mentions": _agg("novelty_mentions"),
        "formula_signals": _agg("formula_signals"),
        "figure_table_refs": _agg("figure_table_refs"),
        "argument_markers_per_1k": _agg("argument_markers_per_1k", lambda v: round(sum(v) / len(v), 2)),
        "proposes_method": any(d["fingerprint"]["proposes_method"] for d in docs),
        "has_innovation_section": any(d["fingerprint"]["has_innovation_section"] for d in docs),
        "ablation_tuning": any(d["fingerprint"]["ablation_tuning"] for d in docs),
        "multi_dataset": any(d["fingerprint"]["multi_dataset"] for d in docs),
        "has_significance": any(d["fingerprint"]["has_significance"] for d in docs),
        "has_multiseed": any(d["fingerprint"]["has_multiseed"] for d in docs),
    }
    names = {d["file"]: d["fingerprint"]["cjk_chars"] for d in docs}
    rubric = (
        "【高质量本科论文参照标尺（来自本地范文库，评阅时据此判断产出质量）】\n"
        f"参照样本：{'、'.join(f'{k}（{v} 汉字）' for k, v in names.items()) or '暂无范文'}。\n"
        f"范文典型水平：正文约 {typical['cjk_chars'] or 0} 汉字、"
        f"{typical['chapter_count'] or 0} 个章节、{typical['ref_count'] or 0} 篇参考文献、"
        f"{typical['formula_signals'] or 0} 处公式信号、{typical['figure_table_refs'] or 0} 处图表引用、"
        f"每千字约 {typical['argument_markers_per_1k'] or 0} 处因果/证据连接词；"
        + ("明确提出新方法、" if typical["proposes_method"] else "")
        + ("设独立创新点小节、" if typical["has_innovation_section"] else "")
        + ("做消融或超参寻优、" if typical["ablation_tuning"] else "")
        + ("跨数据集验证、" if typical["multi_dataset"] else "")
        + ("报告显著性检验。" if typical["has_significance"] else "")
        + "\n评阅要求：把上述特征当作「好论文长什么样」的参照，重点判断论证是否成立、"
          "证据是否支撑结论、章节是否真的承担了论证任务；不要因为模板字段齐全就判为高质量。"
    )
    all_txt = list(Path(corpus_dir).glob("*.txt"))
    skipped = [q.name for q in all_txt if str(q) not in {d["file"] for d in docs}
               and q.name not in {d["file"] for d in docs}
               and str(q) not in {str(Path(corpus_dir) / d["file"]) for d in docs}]
    if skipped:
        rubric += "（范文库另有不可读样本已剔除：" + "、".join(skipped) + "；可往 corpus/ 放入更多高质量中文论文 txt 扩展标尺）"
    return {"schema": "reference-profile/v1", "sources": docs,
            "typical": typical, "rubric_text": rubric,
            "readable_count": len(docs)}


def quality_gap(md: str, corpus_dir: str = "corpus") -> str:
    """网关不可用时的确定性质量差距：只写能从文本数出来的事实，不假装通读过。"""
    fp = extract_fingerprint(md)
    ref = reference_profile(corpus_dir)
    t = ref["typical"]
    lines = ["【范文对标（确定性信号，未做语义通读）】"]
    checks = [
        ("正文篇幅", fp["cjk_chars"], t.get("cjk_chars"), "汉字"),
        ("章节数", fp["chapter_count"], t.get("chapter_count"), "章"),
        ("参考文献", fp["ref_count"], t.get("ref_count"), "篇"),
        ("公式信号", fp["formula_signals"], t.get("formula_signals"), "处"),
        ("图表引用", len(re.findall(r"图\s*\d|表\s*\d|式\s*\(|公式\s*\d", md)), t.get("figure_table_refs"), "处"),
    ]
    for name, cur, refv, unit in checks:
        if refv:
            lines.append(f"- {name}：本文 {cur} {unit}，范文典型 {refv} {unit}"
                         + ("（明显偏少）" if cur < refv * 0.6 else ""))
    for label, key in [("方法原创性", "proposes_method"), ("独立创新点小节", "has_innovation_section"),
                       ("消融/超参寻优", "ablation_tuning"), ("跨数据集验证", "multi_dataset"),
                       ("显著性检验", "has_significance")]:
        if t.get(key) and not fp.get(key):
            lines.append(f"- 缺少范文具备的「{label}」信号")
    lines.append("以上只为可核对的量与信号；论证是否成立、文字是否通顺，需模型通读后判断。")
    return "\n".join(lines)


def benchmark(paper_md_path: str, corpus_dir: str = "corpus") -> dict:
    """对一篇论文做范文对标，返回结构化差距报告。"""
    paper_text = Path(paper_md_path).read_text(encoding="utf-8", errors="ignore")
    paper_fp = extract_fingerprint(paper_text)

    corpus = []
    for cf in _corpus_files(corpus_dir):
        t = Path(cf).read_text(encoding="utf-8", errors="ignore")
        fp = extract_fingerprint(t)
        corpus.append({"file": Path(cf).name, "fingerprint": fp})

    # 范文在各信号上的「典型值」（取各范文最大值，作为质量天花板参照）
    def _agg(key, agg=max):
        vals = [c["fingerprint"].get(key) for c in corpus if c["fingerprint"].get(key) is not None]
        return agg(vals) if vals else None

    corpus_typical = {
        "cjk_chars": _agg("cjk_chars", max),
        "novelty_mentions": _agg("novelty_mentions", max),
        "proposes_method": any(c["fingerprint"]["proposes_method"] for c in corpus),
        "has_innovation_section": any(c["fingerprint"]["has_innovation_section"] for c in corpus),
        "chapter_count": _agg("chapter_count", max),
        "ref_count": _agg("ref_count", max),
        "top_tier_count": _agg("top_tier_count", max),
        "formula_signals": _agg("formula_signals", max),
        "ablation_tuning": any(c["fingerprint"]["ablation_tuning"] for c in corpus),
        "multi_dataset": any(c["fingerprint"]["multi_dataset"] for c in corpus),
        "dataset_scale_max": _agg("dataset_scale_max", max),
        "has_significance": any(c["fingerprint"]["has_significance"] for c in corpus),
        "has_cv": any(c["fingerprint"]["has_cv"] for c in corpus),
        "has_multiseed": any(c["fingerprint"]["has_multiseed"] for c in corpus),
    }

    # 差距点：逐信号对比，标注 paper 是否达到范文典型水平
    gaps = []
    def _gap(name, paper_val, corpus_val, ok_when):
        ok = ok_when(paper_val, corpus_val)
        if not ok:
            gaps.append({"signal": name, "paper": paper_val, "corpus_typical": corpus_val})

    _gap("方法原创性(提出新框架/算法/模型)", paper_fp["proposes_method"],
         corpus_typical["proposes_method"], lambda p, c: bool(p))
    _gap("创新点小节", paper_fp["has_innovation_section"],
         corpus_typical["has_innovation_section"], lambda p, c: bool(p))
    _gap("章节数", paper_fp["chapter_count"], corpus_typical["chapter_count"],
         lambda p, c: (p or 0) >= (c or 0))
    _gap("参考文献条数", paper_fp["ref_count"], corpus_typical["ref_count"],
         lambda p, c: (p or 0) >= (c or 0) * 0.6)  # 达到范文 60% 即算不拉胯
    _gap("顶会/顶刊 venue 数", paper_fp["top_tier_count"], corpus_typical["top_tier_count"],
         lambda p, c: (p or 0) >= max(1, (c or 0) * 0.3))
    _gap("消融/超参寻优", paper_fp["ablation_tuning"], corpus_typical["ablation_tuning"],
         lambda p, c: bool(p) if c else True)
    _gap("跨数据集验证", paper_fp["multi_dataset"], corpus_typical["multi_dataset"],
         lambda p, c: bool(p) if c else True)
    _gap("显著性检验", paper_fp["has_significance"], corpus_typical["has_significance"],
         lambda p, c: bool(p) if c else True)
    _gap("交叉验证", paper_fp["has_cv"], corpus_typical["has_cv"],
         lambda p, c: bool(p) if c else True)
    _gap("多种子复现", paper_fp["has_multiseed"], corpus_typical["has_multiseed"],
         lambda p, c: bool(p) if c else True)
    _gap("公式/数学推导密度", paper_fp["formula_signals"], corpus_typical["formula_signals"],
         lambda p, c: (p or 0) >= max(1, (c or 0) * 0.15))

    # 头号差距：方法原创性（若有）
    headline = None
    if not paper_fp["proposes_method"]:
        headline = ("头号差距在「方法原创性」：范文为提出新框架/算法的原创方法研究，"
                    "本论文为对现有方法的统一基准对比（模型对比型），二者在创新层级上存在本质差距；"
                    "提升路径是在 spec 中加入本文提出的改进方法并配消融/跨数据集验证。")
    elif gaps:
        headline = "总体接近范文框架，但在以下信号上仍有差距：" + "、".join(g["signal"] for g in gaps[:3]) + "。"
    else:
        headline = "本论文在大部分质量指纹上已接近范文典型水平。"

    return {
        "paper_fingerprint": paper_fp,
        "corpus_typical": corpus_typical,
        "corpus_files": [c["file"] for c in corpus],
        "gaps": gaps,
        "headline_gap": headline,
        "gap_count": len(gaps),
    }


def _readable(report: dict) -> str:
    pf = report["paper_fingerprint"]
    ct = report["corpus_typical"]
    lines = []
    lines.append("【范文对标报告】")
    lines.append(f"对标范文：{', '.join(report['corpus_files']) or '（无）'}")
    lines.append("")
    lines.append("信号             | 本论文 | 范文典型")
    lines.append("-----------------|--------|---------")
    rows = [
        ("方法原创性", "是" if pf["proposes_method"] else "否", "是" if ct["proposes_method"] else "否"),
        ("创新点小节", "有" if pf["has_innovation_section"] else "无", "有" if ct["has_innovation_section"] else "无"),
        ("章节数", pf["chapter_count"], ct["chapter_count"]),
        ("参考文献数", pf["ref_count"], ct["ref_count"]),
        ("顶会/顶刊 venue", pf["top_tier_count"], ct["top_tier_count"]),
        ("消融/调优", "有" if pf["ablation_tuning"] else "无", "有" if ct["ablation_tuning"] else "无"),
        ("跨数据集验证", "有" if pf["multi_dataset"] else "无", "有" if ct["multi_dataset"] else "无"),
        ("显著性检验", "有" if pf["has_significance"] else "无", "有" if ct["has_significance"] else "无"),
        ("交叉验证", "有" if pf["has_cv"] else "无", "有" if ct["has_cv"] else "无"),
        ("多种子复现", "有" if pf["has_multiseed"] else "无", "有" if ct["has_multiseed"] else "无"),
        ("公式/推导密度", pf["formula_signals"], ct["formula_signals"]),
        ("样本量上限", pf["dataset_scale_max"], ct["dataset_scale_max"]),
    ]
    for name, p, c in rows:
        lines.append(f"{name:<14} | {str(p):<6} | {c}")
    lines.append("")
    lines.append("头号差距：" + report["headline_gap"])
    if report["gaps"]:
        lines.append("")
        lines.append("其余差距信号：")
        for g in report["gaps"]:
            lines.append(f"  - {g['signal']}：本论文={g['paper']}｜范文典型={g['corpus_typical']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="范文对标：把生成论文与高质量范文做质量指纹对比")
    ap.add_argument("paper", help="待对标的 paper_full.md 路径")
    ap.add_argument("--corpus", default="corpus", help="范文目录（默认 corpus/）")
    ap.add_argument("--out", default="", help="输出 JSON 报告路径（可选）")
    args = ap.parse_args()
    rep = benchmark(args.paper, args.corpus)
    print(_readable(rep))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[benchmark] JSON 报告已写入 {args.out}")


if __name__ == "__main__":
    main()
