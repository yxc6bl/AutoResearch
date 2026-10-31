#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvalAgent —— 质量评分门（升级自 audit_consistency.py）。

落地导师要求：「让 AI 评判抓的文献质量好不好、写出的论文质量好不好」。

六个评分维度：
  1. consistency（真实性硬门）：复用 audit_consistency 的裁决表 vs 结论一致性 +
     素材谎言检测。不过即 fail，是「零编造」底线的守门员。
  2. structure（结构完整度）：对照毕业论文章节 rubric 检查必备章节是否齐全。
  3. authenticity（引用与真实性）：正文 [N] 引用与参考文献条目是否对应、
     论文是否承载真实实验数字。
  4. anti_ai（反 AI 腔）：启发式检测套话、≥3 句同开头排比、单句成段等 AI 写作痕迹。
  5. literature（文献质量）：基于抓取候选的 quality_score / 数量 / 时间跨度评判。
  6. content_quality（内容严谨度）：对照 985 优秀论文标准（清北 / 上交），启发式检查
     文献综述逻辑组织、各章小结、创新点 / 独立见解标注、实验严谨度、
     图表公式规范、参考文献质量与时效——这是 985 与普通毕设的核心差距。

门控规则：硬门（consistency）必须满分，且综合分 ≥ 70，verdict.passed 才为 True。
Orchestrator 据此决定是否打回重做。

CLI：
  python eval_agent.py <paper_full.md> [--lit 02_literature_candidates.json] [--out 07_eval_verdict.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# 保证从任意入口加载时都能 import 根目录的 audit_consistency（单一真相源）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import audit_consistency as audit  # noqa: E402

# 毕业论文章节 rubric：核心必备 vs 推荐（兼容新旧标题，按子串匹配）
CORE_SECTIONS = [
    ("摘要", ["摘要"]),
    ("绪论", ["绪论"]),
    ("国内外研究现状", ["国内外研究现状", "文献综述", "相关工作"]),
    ("研究方法", ["研究方法", "方法", "方法学"]),
    ("实验设置与数据", ["实验设置与数据", "实验设置", "实验"]),
    ("实验结果与分析", ["实验结果与分析", "实验结果", "结果"]),
    ("深入分析与假设检验", ["深入分析与假设检验", "深入分析"]),
    ("讨论", ["讨论"]),
    ("结论与展望", ["结论与展望", "结论"]),
    ("参考文献", ["参考文献"]),
]
REC_SECTIONS = [
    ("研究局限", ["局限性", "研究局限"]),
    ("附录", ["附录"]),
]

# 反 AI 腔套话词（高频 AI 写作痕迹）
AI_CLICHE = [
    "具有重要意义", "被广泛采用", "值得进一步研究", "综上所述",
    "本文的贡献在于", "随着人工智能的发展", "近年来，随着",
    "具有重要的理论意义和实际应用价值", "在一定程度上",
]
# 过度排比常见开头
PARALLEL_PREFIXES = ["本文", "我们", "通过", "首先", "其次", "基于", "实验表明", "结果表明"]


def check_structure(md: str):
    """对照 rubric 检查章节完整度。"""
    heads = " ".join(re.findall(r'^#{1,4}\s+(.*)$', md, re.M))
    missing_core = [name for name, als in CORE_SECTIONS if not any(a in heads for a in als)]
    missing_rec = [name for name, als in REC_SECTIONS if not any(a in heads for a in als)]
    score = 100 - len(missing_core) * 11 - min(10, len(missing_rec) * 5)
    return max(0, score), missing_core, missing_rec


def check_references(md: str):
    """参考文献与正文 [N] 引用对应 + 真实数字承载。"""
    issues = []
    cites = set(int(x) for x in re.findall(r'\[(\d+)\]', md))
    m = re.search(r'#{1,4}\s*参考文献', md)
    ref_block = md[m.start():] if m else ""
    ref_nums = set(int(x) for x in re.findall(r'^\[(\d+)\]', ref_block, re.M))
    if cites and ref_nums:
        missing = sorted(cites - ref_nums)
        if missing:
            issues.append(f"正文引用了参考文献中不存在的编号：{missing}")
        ghost = sorted(ref_nums - cites)
        if ghost:
            issues.append(f"参考文献列表中存在未被正文引用的条目：{ghost}")
    elif cites and not ref_nums:
        issues.append("正文有 [N] 引用，但找不到参考文献章节")
    if not re.search(r'\d+\.\d+%?|\d+%', md):
        issues.append("全文未检测到任何具体数值（真实实验论文应承载数字）")
    score = 100
    if any('不存在' in i or '找不到' in i for i in issues):
        score -= 40
    if any('未被正文引用' in i for i in issues):
        score -= 20
    if any('未检测到任何具体数值' in i for i in issues):
        score -= 20
    return max(0, score), issues


def check_anti_ai(md: str):
    """启发式反 AI 腔检测。"""
    markers = []
    cliche_hits = {c: md.count(c) for c in AI_CLICHE if md.count(c) > 0}
    for c, n in cliche_hits.items():
        markers.append(f"套话「{c}」出现 {n} 次")
    # 只统计「正文段落」，排除标题 / 表格 / 列表 / 引用 / 图片 / 加粗行等非段落结构，
    # 否则一篇结构正常的论文会因大量「第 N 章」标题被误判为单句成段。
    def _is_prose(p: str) -> bool:
        s = p.strip()
        if not s:
            return False
        first = s.splitlines()[0].strip()
        if first.startswith(("#", "|", ">", "!", "*", "-")):
            return False
        if re.match(r"^\d+\.\s", first):
            return False
        return True
    paras = [p for p in re.split(r'\n{2,}', md) if _is_prose(p)]
    parallel = 0
    for p in paras:
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        if len(lines) >= 3:
            pref = [next((pr for pr in PARALLEL_PREFIXES if l.startswith(pr)), None) for l in lines]
            cnt = Counter(p for p in pref if p)
            if cnt and cnt.most_common(1)[0][1] >= 3:
                parallel += 1
    if parallel:
        markers.append(f"{parallel} 处段落出现≥3句相同开头的排比堆砌")
    single = sum(1 for p in paras if len([l for l in p.splitlines() if l.strip()]) == 1)
    ratio = single / max(1, len(paras))
    if ratio > 0.3:
        markers.append(f"单句成段比例 {ratio:.0%}（偏高，AI 腔特征）")
    score = 100 - sum(n * 5 for n in cliche_hits.values()) - parallel * 10
    score -= max(0, (ratio - 0.3)) * 100
    return max(0, min(100, score)), markers


def check_literature(candidates):
    """基于抓取候选评判文献质量。"""
    if not candidates:
        return 0, ["无文献候选数据，无法评判（请先运行 LiteratureAgent）"], "n/a", {}
    n = len(candidates)
    qs = [c.get("quality_score", 0) or 0 for c in candidates]
    # quality_score 为被引数+时效性算出的 0~1 归一值，故高质量阈值取 0.5，
    # 而非 1.0（旧写法使 high_quality 恒为 0，统计失真）。
    high = sum(1 for q in qs if q >= 0.5)
    avg = sum(qs) / len(qs) if qs else 0
    years = [int(c.get("year", 0)) for c in candidates if str(c.get("year", "")).isdigit()]
    span = (max(years) - min(years)) if years else 0
    issues = []
    if n < 8:
        issues.append(f"文献数量仅 {n} 篇（本科毕设通常需 ≥15 篇）")
    if qs and max(qs) < 0.5:
        issues.append("最高质量分偏低，缺乏高被引/权威文献")
    if years and span < 3:
        issues.append(f"文献时间跨度仅 {span} 年，缺乏年代纵深")
    score = 100
    if n < 8:
        score -= 30
    elif n < 15:
        score -= 10
    score -= (1 - avg) * 40
    if years and span < 3:
        score -= 10
    grade = "good" if score >= 75 else ("acceptable" if score >= 50 else "weak")
    return max(0, min(100, score)), issues, grade, {"count": n, "high_quality": high, "avg_q": round(avg, 2), "span": span}


def _slice_chapter(md: str, start_hint: str, end_hints):
    """截取从含 start_hint 的标题到下一个含 end_hints 任一标题之间的正文。"""
    lines = md.splitlines()
    start = None
    for i, l in enumerate(lines):
        if re.match(r'^#{1,4}\s+.*' + re.escape(start_hint), l):
            start = i
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        for h in end_hints:
            if re.match(r'^#{1,4}\s+.*' + re.escape(h), lines[j]):
                end = j
                break
        if end != len(lines):
            break
    return "\n".join(lines[start:end])


def check_content_quality(md: str):
    """对照 985 优秀论文标准（清北 / 上交）的启发式内容严谨度检查。

    这是 985 级与普通毕设的真正差距所在：不在于格式，而在于论证深度与严谨度。
    涵盖六项：文献综述逻辑组织、各章小结、创新点标注、实验严谨度、图表公式规范、
    参考文献质量与时效。
    """
    issues = []

    # 第2章 国内外研究现状
    related = _slice_chapter(md, "国内外研究现状", ["研究方法"])
    rel_cites = len(re.findall(r'\[\d+\]', related)) if related else 0
    rel_oriented = bool(re.search(r'本文|本文提出|本文与|差异|不同|局限', related or ""))
    if rel_cites < 3 or not rel_oriented:
        issues.append("文献综述疑似流水账（引用<3 或 未定位本文差异）")

    # 各章小结
    xj = md.count("小结")
    if xj < 2:
        issues.append(f"仅 {xj} 处「小结」，缺少各章小结")

    # 创新点 / 独立见解（侧重结论章）
    concl = _slice_chapter(md, "结论与展望", ["参考文献"])
    innov = bool(re.search(r'创新|独立见解|本文提出|本文贡献|本文设计', concl or md))
    if not innov:
        issues.append("结论未明确创新点 / 独立见解，或未区分本人与他人工作")

    # 实验严谨度
    exp = _slice_chapter(md, "实验设置与数据", ["实验结果与分析"]) or ""
    exp += _slice_chapter(md, "实验结果与分析", ["深入分析与"]) or ""
    rigor = bool(re.search(r'基线|baseline|方差|种子|交叉验证|消融|对比', exp or "", re.I))
    if not rigor:
        issues.append("实验叙述缺对比基线 / 稳健性 / 消融分析")

    # 图表公式规范
    fig_ok = bool(re.search(r'如图\s*\d|表\s*\d|如下表|图\s*\d', md))
    eq_ok = bool(re.search(r'式\s*[（(]\d|公式', md))
    if not (fig_ok and eq_ok):
        issues.append("图表 / 公式编号引用不规范或缺失")

    # 参考文献质量与时效
    cites_total = len(set(int(x) for x in re.findall(r'\[(\d+)\]', md)))
    if cites_total < 20:
        issues.append(f"参考文献引用总数 {cites_total}（985 通常 ≥20）")
    years = [int(y) for y in re.findall(r'\b(?:19|20)\d{2}\b', md)]
    recent5 = sum(1 for y in years if 2021 <= y <= 2026)
    recent2 = sum(1 for y in years if 2025 <= y <= 2026)
    if cites_total >= 20 and years and (recent5 < 5 or recent2 < 1):
        issues.append("参考文献缺乏近五年 / 近两年文献")

    score = 100 - len(issues) * 12
    return max(0, min(100, score)), issues


def run(paper_path, lit_candidates=None, store=None, out_path=None):
    """主入口：对一篇论文给出评分门 verdict。"""
    md = Path(paper_path).read_text(encoding="utf-8")
    verdict = {"passed": False, "gate": "hard_fail", "scores": {}, "issues": [], "suggestions": []}

    # 1) 真实性硬门（复用 audit）
    tbl = audit.parse_verdict_table(md)
    concl = audit.parse_conclusion(md)
    lies = audit.detect_lies(md)
    hard = []
    all_h = sorted(set(tbl) | set(concl), key=lambda k: int(re.sub(r'\D', '', k) or 0))
    for h in all_h:
        tv, cv = (tbl.get(h) or "").strip(), (concl.get(h) or "").strip()
        if tv and cv and tv != cv:
            hard.append(f"{h} 不一致：裁决表='{tv}' vs 结论='{cv}'")
    for l in lies:
        hard.append("疑似素材谎言：" + l)
    consistency = 100 if not hard else 0
    verdict["scores"]["consistency"] = consistency
    for h in hard:
        verdict["issues"].append("真实性硬门：" + h)

    # 2) 结构完整度
    st, mc, mr = check_structure(md)
    verdict["scores"]["structure"] = st
    if mc:
        verdict["issues"].append("缺失核心章节：" + "、".join(mc))
        verdict["suggestions"].append("补齐缺失核心章节（" + "、".join(mc) + "）以达到毕业论文章节水平")
    if mr:
        verdict["suggestions"].append("建议补充推荐章节：" + "、".join(mr))

    # 3) 引用与真实性
    au, auis = check_references(md)
    verdict["scores"]["authenticity"] = au
    for i in auis:
        verdict["issues"].append("引用/真实性：" + i)
    if any('未检测到任何具体数值' in i for i in auis):
        verdict["suggestions"].append("确保论文承载真实实验数字，引用分析引擎产出")
    if any('不存在' in i or '找不到' in i for i in auis):
        verdict["suggestions"].append("修正参考文献与正文引用编号的对应关系")

    # 4) 反 AI 腔
    ai, aim = check_anti_ai(md)
    verdict["scores"]["anti_ai"] = ai
    if aim:
        verdict["issues"].append("AI 腔：" + "；".join(aim))
        verdict["suggestions"].append("降低套话与排比堆砌，增加具体论证与口语化连接，提升反 AI 腔")

    # 5) 文献质量
    cands = lit_candidates
    if cands is None and store is not None:
        cands = _collect_candidates(store)
    lits, litis, litg, litstat = check_literature(cands or [])
    verdict["scores"]["literature"] = lits
    verdict["literature_grade"] = litg
    verdict["literature_stat"] = litstat
    for i in litis:
        verdict["issues"].append("文献：" + i)
        if "数量" in i:
            verdict["suggestions"].append("增加文献抓取量与质量（已支持 arxiv+OpenAlex 多源与 quality_score）")

    # 6) 内容严谨度（985 级门槛）
    cq, cqis = check_content_quality(md)
    verdict["scores"]["content_quality"] = cq
    for i in cqis:
        verdict["issues"].append("内容严谨度：" + i)
    if any('流水账' in i or '小结' in i for i in cqis):
        verdict["suggestions"].append("文献综述按主题组织并定位本文差异；各章补「本章小结」")
    if any('创新点' in i or '实验叙述' in i for i in cqis):
        verdict["suggestions"].append("结论明确创新点 / 独立见解；实验补充对比基线与稳健性证据")

    # 综合与门控
    overall = round(0.35 * consistency + 0.15 * st + 0.1 * au + 0.1 * ai + 0.1 * lits + 0.2 * cq, 1)
    verdict["overall"] = overall
    verdict["scores"]["overall"] = overall
    hard_ok = consistency == 100 and not hard
    verdict["gate"] = "hard_pass" if hard_ok else "hard_fail"
    verdict["passed"] = hard_ok and overall >= 70
    if not hard_ok:
        verdict["suggestions"].insert(0, "硬门未通过（一致性/谎言），必须修正后重做")
    if out_path:
        Path(out_path).write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    return verdict


def _collect_candidates(store):
    """从 MemoryStore 的 literature 命名空间聚合所有已记忆文献。"""
    try:
        return store.query("literature", {})
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description="EvalAgent 质量评分门")
    ap.add_argument("paper", help="paper_full.md 路径")
    ap.add_argument("--lit", help="文献候选 JSON（02_literature_candidates.json）路径")
    ap.add_argument("--out", help="输出 verdict JSON 路径")
    args = ap.parse_args()
    cands = None
    if args.lit and Path(args.lit).exists():
        try:
            d = json.loads(Path(args.lit).read_text(encoding="utf-8"))
            cands = d if isinstance(d, list) else (d.get("papers") or d.get("candidates") or [])
        except Exception:
            cands = None
    v = run(args.paper, lit_candidates=cands, out_path=args.out)
    print(json.dumps(v, ensure_ascii=False, indent=2))
    sys.exit(0 if v["passed"] else 1)


if __name__ == "__main__":
    main()
