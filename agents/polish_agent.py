#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""润色 Agent（PolishAgent）—— 独立于写作的可选步骤。

为什么单独拆出来：写作负责「把证据组织成论文」，润色负责「把论文改成读得顺的
书面表达」。两者失败原因不同、重跑成本也不同（写作要调模型，润色纯规则），
混在一起会导致「只想改文风却要重跑整篇」。

本 Agent 只做**确定性、可解释**的改动，不调模型：
  1) 去套话：删除「值得注意的是」「综上所述」「总的来说」等空转连接词
  2) 标点规范：半角逗号/句号转全角（仅中文语境）、合并重复标点、去句末多余空格
  3) 空格规范：中英文/数字之间补一个空格；删除行首多余缩进
  4) 单句成段合并：仅当相邻两段都很短（<25 字）且无标题/列表/表格时合并

每条改动都计数并写入 polish_report.json，便于回退与核对——不做「悄悄改写」。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 空转连接词：删掉不影响语义，但会让文章显得像模板
_CLICHE = [
    "值得注意的是，", "值得注意的是", "综上所述，", "综上所述",
    "总的来说，", "总的来说", "总而言之，", "总而言之",
    "不难发现，", "不难看出，", "显而易见，", "需要指出的是，",
    "换言之，", "也就是说，", "换句话说，",
]

_CJK = r"\u4e00-\u9fff"


def _strip_cliche(text: str) -> tuple[str, int]:
    n = 0
    for w in _CLICHE:
        c = text.count(w)
        if c:
            text = text.replace(w, "")
            n += c
    return text, n


def _cjk_ratio(line: str) -> float:
    """中文占比：用来判断这一行该用中文标点还是英文标点。"""
    letters = [c for c in line if c.isalpha() or "\u4e00" <= c <= "\u9fff"]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\u4e00" <= c <= "\u9fff") / len(letters)


def _fix_punct(text: str) -> tuple[str, int]:
    """中文标点规范：**逐行判断语种**，中文行才转全角。

    早期版本全局替换，把英文摘要里的句点也改成了「。」（因为句中出现中文方法名），
    导致中英混排段落标点错乱。现在只处理中文占比 >50% 的行。
    """
    n = 0
    out_lines = []
    for line in text.split("\n"):
        if _cjk_ratio(line) <= 0.5:
            out_lines.append(line)
            continue

        def _cn_comma(m):
            nonlocal n
            n += 1
            return "，"

        def _cn_period(m):
            nonlocal n
            n += 1
            return "。"

        line = re.sub(rf"(?<=[{_CJK}]),", _cn_comma, line)
        line = re.sub(rf"(?<=[{_CJK}])\.(?=\s|$)", _cn_period, line)
        for pat, rep in [(r"，{2,}", "，"), (r"。{2,}", "。"), (r"；{2,}", "；"),
                         (r"、{2,}", "、"), (r"！{2,}", "！"), (r"？{2,}", "？")]:
            line, k = re.subn(pat, rep, line)
            n += k
        out_lines.append(line)
    text = "\n".join(out_lines)
    text, k = re.subn(r"[ \t]+(\n|$)", r"\1", text)
    n += k
    return text, n


def _fix_spacing(text: str) -> tuple[str, int]:
    n = 0
    # 中文与英文/数字之间补空格（跳过已有空格与标点）
    def _add(m):
        nonlocal n
        n += 1
        return f"{m.group(1)} {m.group(2)}"
    # 只处理「中文 ↔ 英文字母」：数字不加空格（否则会把「第1章」「8个样本」改坏），
    # 也不动「表1」「图2」「式(3)」这类编号
    text = re.sub(rf"([{_CJK}])([A-Za-z])", _add, text)
    text = re.sub(rf"([A-Za-z])([{_CJK}])", _add, text)
    # 行首多余缩进（保留标题与列表结构）
    lines = []
    for line in text.split("\n"):
        if line.startswith(("#", "|", "-", "*", ">", "!", "```")) or not line.strip():
            lines.append(line)
        else:
            lines.append(line.lstrip(" \t\u3000"))
    return "\n".join(lines), n


def _merge_short_paragraphs(text: str, min_len: int = 25) -> tuple[str, int]:
    """把相邻的两个超短段落合并（只处理纯正文段落，不碰标题/表格/列表/图片）。"""
    blocks = text.split("\n\n")
    out: list[str] = []
    merged = 0
    skip_prefix = ("#", "|", "-", "*", ">", "!", "```", "**表", "**图")
    for b in blocks:
        s = b.strip()
        if (out and s and out[-1].strip()
                and not s.startswith(skip_prefix)
                and not out[-1].strip().startswith(skip_prefix)
                and len(s) < min_len
                and len(out[-1].strip()) < min_len
                and not out[-1].strip().endswith(("：", ":"))):
            out[-1] = out[-1].rstrip() + s
            merged += 1
        else:
            out.append(b)
    return "\n\n".join(out), merged


def run(paper_md, out_path=None, min_paragraph: int = 25) -> dict:
    """对论文 Markdown 做确定性润色，返回统计；不覆盖原文件，另存 *_polished.md。"""
    src = Path(paper_md)
    text = src.read_text(encoding="utf-8")
    stats = {}
    text, stats["去套话"] = _strip_cliche(text)
    text, stats["标点规范"] = _fix_punct(text)
    text, stats["空格规范"] = _fix_spacing(text)
    text, stats["短段合并"] = _merge_short_paragraphs(text, min_paragraph)

    dst = Path(out_path) if out_path else src.with_name(src.stem + "_polished.md")
    dst.write_text(text, encoding="utf-8")
    report = {"source": str(src), "output": str(dst), "changes": stats,
              "total": sum(stats.values())}
    (dst.parent / "polish_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PolishAgent] {src.name} → {dst.name}｜改动 {report['total']} 处：{stats}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="润色 Agent（确定性，不调模型）")
    ap.add_argument("--md", required=True, help="paper_full.md 路径")
    ap.add_argument("--out", default="", help="输出路径（默认 *_polished.md）")
    ap.add_argument("--min-paragraph", type=int, default=25, help="短段合并阈值（字数）")
    a = ap.parse_args()
    run(a.md, a.out or None, a.min_paragraph)
