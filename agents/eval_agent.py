#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvalAgent —— 质量评分门（升级自 audit_consistency.py）。

落地导师要求：「让 AI 评判抓的文献质量好不好、写出的论文质量好不好」。

评分标准**有据可查**：维度、权重、等级阈值均来自清北及多所 985 高校公开的
本科毕设/毕业论文评分办法，详见 docs/EVAL_RUBRIC.md（清华 / 哈工大(威海) /
北大 / 中科大 / 东北大学 / 武大 / 辽大）。本文件只做「把公认的 rubric 映射为
可机器执行的确定性检查」，不自行发明标准。

六维评分（权重来源见 docs/EVAL_RUBRIC.md 第二节）：
  1. topic_sig     选题意义            权重 12%
  2. literature    文献调研            权重 15%
  3. structure     逻辑构建            权重 10%
  4. methodology   专业能力·方法与内容 权重 33%
  5. innovation    创新               权重 15%
  6. writing_norms 撰写质量与学术规范 权重 15%

零编造硬门（独立，不计入加权）：consistency 必须满分，否则一律不及格重做。

等级（五级分制，见 docs/EVAL_RUBRIC.md 第三节）：
  优秀 ≥90 / 良好 80–89 / 中等 70–79 / 及格 60–69 / 不及格 <60。
  通过判定 = consistency 硬门满分 且 overall ≥ 60。

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

# ── 权重与等级（溯源见 docs/EVAL_RUBRIC.md） ──────────────────────────
WEIGHTS = {
    "topic_sig": 0.12,       # 选题意义 — 哈工大10% / 东北大学18% / 清华选题好 / 北大①
    "literature": 0.15,      # 文献调研 — 哈工大调研论证20% / 武大15% / 东北应用文献9%
    "structure": 0.10,       # 逻辑构建 — 中科大逻辑构建 / 清华结构严谨 / 北大③
    "methodology": 0.33,     # 专业能力·方法与内容 — 哈工大35% / 东北能力水平52% / 中科大专业能力
    "innovation": 0.15,      # 创新 — 哈工大创新与成效25% / 北大⑤ / 中科大创新与特色
    "writing_norms": 0.15,   # 撰写质量与学术规范 — 哈工大10% / 辽大学术规范30% / 北大④⑥⑦
}
# 零编造硬门：独立，不计入加权（真实 rubric 默认论文真实，本系统额外用机器兜底）
GRADE_BANDS = [("优秀", 90), ("良好", 80), ("中等", 70), ("及格", 60)]  # <60 不及格


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
    """对照 rubric 检查章节完整度（逻辑构建维度）。"""
    heads = " ".join(re.findall(r'^#{1,4}\s+(.*)$', md, re.M))
    missing_core = [name for name, als in CORE_SECTIONS if not any(a in heads for a in als)]
    missing_rec = [name for name, als in REC_SECTIONS if not any(a in heads for a in als)]
    score = 100 - len(missing_core) * 11 - min(10, len(missing_rec) * 5)
    return max(0, score), missing_core, missing_rec


def check_references(md: str):
    """参考文献与正文 [N] 引用对应 + 真实数字承载（计入 methodology 维度）。"""
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
    """启发式反 AI 腔检测（计入 writing_norms 维度）。"""
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
    """基于抓取候选评判文献质量（计入 literature 维度）。"""
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


def check_topic_significance(md: str):
    """选题意义维度：绪论是否陈述「现实/理论相关性」与「研究空白」。

    依据：清华「选题好、有现实意义」；北大①选题是否具有理论或实践意义；
    哈工大选题(10%)「有实际应用背景与可操作性，具备理论和现实意义」；
    东北大学选题质量18%「理论意义或实际意义」。
    判定用信号词而非字面「研究意义」四字，避免误伤（如本文绪论以
    "乳腺癌早期筛查依赖…" 表相关性、"尚缺乏…严格量化比较" 表空白）。
    """
    intro = _slice_chapter(md, "第1章 绪论", ["第2章", "国内外研究现状"])
    if not intro:
        intro = _slice_chapter(md, "绪论", ["国内外研究现状", "研究方法"])
    text = intro or ""
    relevance = bool(re.search(r'临床|现实|实际|应用|需求|意义|价值|重要|背景|迫切|关键', text))
    gap = bool(re.search(r'缺乏|尚未|不足|未[能]?|差距|问题|挑战|难点|瓶颈|少见|鲜[有见]', text))
    issues = []
    if relevance and gap:
        score = 100
    elif relevance or gap:
        score = 60
        issues.append("绪论仅部分体现选题意义（相关性/研究空白不全）")
    else:
        score = 0
        issues.append("绪论未陈述研究的现实/理论意义，也未指出研究空白")
    return score, issues


def check_methodology(md: str):
    """专业能力·方法与内容维度（权重最高 33%）。

    依据：哈工大研究方法及内容(35%)「研究方案科学合理；数据来源可靠，
    分析论证设计正确，结论可信」；东北大学能力水平52%；中科大专业能力。
    综合三项：实验严谨度 + 真实数字承载 + 引用对应。
    """
    issues = []
    exp = _slice_chapter(md, "实验设置与数据", ["实验结果与分析"]) or ""
    exp += _slice_chapter(md, "实验结果与分析", ["深入分析与"]) or ""
    exp += _slice_chapter(md, "深入分析与", ["讨论", "结论"]) or ""
    rigor = bool(re.search(r'基线|baseline|方差|种子|交叉验证|消融|对比|配对|显著性', exp or "", re.I))
    if not rigor:
        issues.append("实验叙述缺对比基线/稳健性/消融或显著性检验")
    realnum = bool(re.search(r'\d+\.\d+%?|\d+%', md))
    if not realnum:
        issues.append("全文未检测到任何具体数值（真实实验论文应承载数字）")
    ref_score, ref_issues = check_references(md)
    for i in ref_issues:
        issues.append("引用/真实性：" + i)
    # 组合：实验严谨 35 + 真实数字 30 + 引用对应 35
    score = 100
    if not rigor:
        score -= 35
    if not realnum:
        score -= 30
    if any('不存在' in i or '找不到' in i for i in ref_issues):
        score -= 25
    if any('未被正文引用' in i for i in ref_issues):
        score -= 10
    return max(0, score), issues


def check_innovation(md: str):
    """创新维度。依据：哈工大创新与成效(25%)；北大⑤观点新颖；中科大创新与特色。

    结论章应明确创新点/独立见解/本文贡献，并区分本人与他人工作。
    """
    concl = _slice_chapter(md, "结论与展望", ["参考文献", "致谢"]) or md
    has_innov = bool(re.search(r'创新|独立见解|本文提出|本文贡献|本文设计|本文首次|新颖', concl))
    has_section = bool(re.search(r'创新点|本文贡献|独立见解', md))
    issues = []
    if not has_innov:
        issues.append("结论未明确创新点 / 独立见解，或未区分本人与他人工作")
    elif not has_section:
        issues.append("结论提及贡献但缺少独立的「创新点」小节，建议单列")
    score = 100 if (has_innov and has_section) else (60 if has_innov else 0)
    return max(0, score), issues


def check_writing_norms(md: str):
    """撰写质量与学术规范维度（权重 15%）。

    依据：哈工大撰写质量(10%)；辽大学术规范30%（文字表达/格式/诚信）；
    北大④文字准确流畅 ⑥引用注释规范 ⑦无抄袭。
    组合：反 AI 腔 50 + 图表公式规范 30 + 参考文献时效 20。
    """
    ai, ai_markers = check_anti_ai(md)
    fig_ok = bool(re.search(r'如图\s*\d|表\s*\d|如下表|图\s*\d|见[图表]', md))
    eq_ok = bool(re.search(r'式\s*[（(]\d|公式', md))
    format_score = 100 if (fig_ok and eq_ok) else 60
    fmt_issue = [] if (fig_ok and eq_ok) else ["图表 / 公式编号引用不规范或缺失"]
    cites_total = len(set(int(x) for x in re.findall(r'\[(\d+)\]', md)))
    years = [int(y) for y in re.findall(r'\b(?:19|20)\d{2}\b', md)]
    recent5 = sum(1 for y in years if 2021 <= y <= 2026)
    recent2 = sum(1 for y in years if 2025 <= y <= 2026)
    if cites_total < 20:
        recency_score = 40
        rec_issue = [f"参考文献引用总数 {cites_total}（985 通常 ≥20）"]
    else:
        recency_score = 100 if (recent5 >= 5 and recent2 >= 1) else 70
        rec_issue = [] if recency_score == 100 else ["参考文献缺乏近五年 / 近两年文献"]
    # 深技大格式合规（关键词/摘要/字数/前置/外文文献）并入撰写规范维度
    fmt_norm, fmt_norm_issues = check_format_compliance(md)
    score = round(0.45 * ai + 0.25 * format_score + 0.15 * recency_score + 0.15 * fmt_norm)
    issues = list(ai_markers) + fmt_issue + rec_issue + fmt_norm_issues
    return max(0, min(100, score)), issues


def _cn_sentences(md: str):
    """提取中文叙述性句子（≥12字），排除标题/表格/列表/代码/引用/参考文献条目。"""
    out = []
    for p in re.split(r'\n{2,}', md):
        s = p.strip()
        if not s:
            continue
        if s.startswith(('#', '|', '>', '!', '*', '-', '```')):
            continue
        if re.match(r'^\d+\.\s', s):
            continue
        if re.match(r'^\[\d+\]', s):  # 参考文献条目
            continue
        for seg in re.split(r'(?<=[。！？；])', s):
            seg = seg.strip()
            if len(seg) >= 12:
                out.append(seg)
    return out


def _cjk4grams(text: str):
    chars = [c for c in text if '一' <= c <= '鿿']
    if len(chars) < 4:
        return set()
    return set(''.join(chars[i:i + 4]) for i in range(len(chars) - 3))


def check_duplication(md: str, lit_candidates=None):
    """本地查重预估（仅本地自检，不含知网全网库）。

    深技大规定：知网文字重合率 <30% 合格，校级优秀 <20%。本函数无法直接调
    知网，故用三路可控信号估算「论文自身重复 + 与参考文献标题逐字复刻」的重合率，
    作为提交前自检。信号：
      (1) 自重复：论文内两句话 4-gram Jaccard≥0.7 且长度接近 → 判定近重复；
      (2) 标题复刻：论文章节句与参考文献标题 4-gram Jaccard≥0.8 且重叠≥8字。
    返回 estimated_rate(0~1)、self_rate、title_rate、样本与告警级别。
    """
    sents = _cn_sentences(md)
    if len(sents) < 5:
        return {"estimated_rate": 0.0, "self_rate": 0.0, "title_rate": 0.0,
                "samples": [], "flag": None}
    grams = [_cjk4grams(s) for s in sents]
    dup, flagged, samples = 0, set(), []
    for i in range(len(sents)):
        if i in flagged or not grams[i]:
            continue
        for j in range(i + 1, len(sents)):
            if j in flagged or not grams[j]:
                continue
            inter = len(grams[i] & grams[j])
            union = len(grams[i] | grams[j])
            if not union:
                continue
            if inter / union >= 0.7:
                li, lj = len(sents[i]), len(sents[j])
                if abs(li - lj) <= max(5, 0.3 * max(li, lj)):
                    flagged.add(j)
                    dup += 1
                    if len(samples) < 5:
                        samples.append(sents[j][:38])
                    break
    self_rate = dup / len(sents)

    titles = [c.get('title', '') for c in (lit_candidates or [])
              if isinstance(c, dict) and c.get('title')]
    title_rate = 0.0
    if titles:
        tgrams = [_cjk4grams(t) for t in titles]
        overlap = 0
        for g in grams:
            if not g:
                continue
            for tg in tgrams:
                if not tg:
                    continue
                inter = len(g & tg)
                union = len(g | tg)
                if union and inter / union >= 0.8 and inter >= 8:
                    overlap += 1
                    break
        title_rate = overlap / len(sents)

    est = min(1.0, self_rate + 0.5 * title_rate)
    flag = None
    if est >= 0.30:
        flag = "fail(≥30% 合格线)"
    elif est >= 0.20:
        flag = "warn(≥20% 校级优秀线)"
    return {"estimated_rate": round(est, 4), "self_rate": round(self_rate, 4),
            "title_rate": round(title_rate, 4), "samples": samples, "flag": flag}


def check_format_compliance(md: str):
    """深技大格式合规（计入撰写规范维度）：关键词 3-4个且每词≤6字、中文摘要≥200字、
    全文中文≥12000字、必备前置「封面」「诚信声明」、外文文献≥2篇。"""
    issues = []
    score = 100
    # 关键词
    m = re.search(r'关键词[：:]\s*(.+)', md)
    if m:
        kws = [k.strip().strip('*').strip('`').strip() for k in re.split(r'[；;]', m.group(1)) if k.strip()]
        if len(kws) > 4:
            issues.append(f"关键词 {len(kws)} 个（深技大要求 3-4 个）")
            score -= 15
        over = [k for k in kws if len(k) > 6]
        if over:
            issues.append(f"关键词超长：{over}（每词应≤6字）")
            score -= 10
    else:
        issues.append("未找到关键词")
        score -= 20
    # 中文摘要字数
    am = re.search(r'#+\s*摘要\s*\n(.*?)\n#+\s*Abstract', md, re.S)
    if am:
        cn = sum(1 for c in am.group(1) if '一' <= c <= '鿿')
        if cn < 200:
            issues.append(f"中文摘要 {cn} 字（深技大要求≥200字）")
            score -= 15
    # 全文字数
    cn_all = sum(1 for c in md if '一' <= c <= '鿿')
    if cn_all < 12000:
        issues.append(f"全文中文 {cn_all} 字（深技大要求≥12000字）")
        score -= 15
    # 必备前置
    if '诚信声明' not in md:
        issues.append("缺少「诚信声明」章节（深技大必备前置）")
        score -= 10
    if '封面' not in md:
        issues.append("缺少「封面」要素（深技大必备前置）")
        score -= 10
    # 外文文献≥2
    mr = re.search(r'#{1,4}\s*参考文献', md)
    ref = md[mr.start():] if mr else ""
    ref_items = re.findall(r'^\[\d+\].*$', ref, re.M)
    foreign = sum(1 for it in ref_items if re.search(r'[A-Za-z]{4,}', it))
    if foreign < 2:
        issues.append(f"外文文献 {foreign} 篇（深技大要求≥2篇）")
        score -= 10
    return max(0, score), issues


def _review_organization(md: str):
    """文献综述逻辑组织（计入 literature 维度）。"""
    related = _slice_chapter(md, "国内外研究现状", ["研究方法", "第3章"])
    rel_cites = len(re.findall(r'\[\d+\]', related)) if related else 0
    rel_oriented = bool(re.search(r'本文|本文提出|本文与|差异|不同|局限', related or ""))
    ok = (rel_cites >= 3 and rel_oriented)
    issue = [] if ok else ["文献综述疑似流水账（引用<3 或 未定位本文差异）"]
    score = 100 if ok else 60
    return score, issue


def run(paper_path, lit_candidates=None, store=None, out_path=None):
    """主入口：对一篇论文给出评分 verdict。"""
    md = Path(paper_path).read_text(encoding="utf-8")
    verdict = {"passed": False, "gate": "hard_fail", "scores": {}, "issues": [], "suggestions": []}

    # 0) 零编造硬门（复用 audit，独立于 rubric 加权）
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

    # 1) 选题意义
    ts, ts_is = check_topic_significance(md)
    verdict["scores"]["topic_sig"] = ts
    for i in ts_is:
        verdict["issues"].append("选题意义：" + i)
        verdict["suggestions"].append("绪论显式陈述研究的现实/理论意义，并指出具体研究空白")

    # 2) 文献调研 = 候选质量 ×0.6 + 综述组织 ×0.4
    cands = lit_candidates
    if cands is None and store is not None:
        cands = _collect_candidates(store)
    litq, litq_is, litg, litstat = check_literature(cands or [])
    rev_s, rev_is = _review_organization(md)
    literature = round(0.6 * litq + 0.4 * rev_s)
    verdict["scores"]["literature"] = literature
    verdict["literature_grade"] = litg
    verdict["literature_stat"] = litstat
    for i in litq_is + rev_is:
        verdict["issues"].append("文献：" + i)
        if "数量" in i or "流水账" in i:
            verdict["suggestions"].append("增加文献抓取量与质量；综述按主题组织并定位本文差异")

    # 3) 逻辑构建
    st, mc, mr = check_structure(md)
    verdict["scores"]["structure"] = st
    if mc:
        verdict["issues"].append("缺失核心章节：" + "、".join(mc))
        verdict["suggestions"].append("补齐缺失核心章节（" + "、".join(mc) + "）以达到毕业论文章节水平")
    if mr:
        verdict["suggestions"].append("建议补充推荐章节：" + "、".join(mr))

    # 4) 专业能力·方法与内容
    mth, mth_is = check_methodology(md)
    verdict["scores"]["methodology"] = mth
    for i in mth_is:
        verdict["issues"].append("专业能力：" + i)
        if "基线" in i or "显著性" in i:
            verdict["suggestions"].append("实验补充对比基线与稳健性/显著性证据")
        if "未检测" in i:
            verdict["suggestions"].append("确保论文承载真实实验数字")
        if "不存在" in i or "找不到" in i:
            verdict["suggestions"].append("修正参考文献与正文引用编号的对应关系")

    # 5) 创新
    inn, inn_is = check_innovation(md)
    verdict["scores"]["innovation"] = inn
    for i in inn_is:
        verdict["issues"].append("创新：" + i)
        verdict["suggestions"].append("结论明确创新点 / 独立见解，建议单列「本文创新点」小节")

    # 6) 撰写质量与学术规范
    wn, wn_is = check_writing_norms(md)
    verdict["scores"]["writing_norms"] = wn
    for i in wn_is:
        verdict["issues"].append("撰写规范：" + i)
        if "套话" in i or "单句成段" in i or "排比" in i:
            verdict["suggestions"].append("降低套话与排比堆砌，提升反 AI 腔")
        if "图表" in i or "公式" in i:
            verdict["suggestions"].append("补充图表 / 公式编号引用，符合学术规范")
        if "参考文献" in i:
            verdict["suggestions"].append("补充近五年 / 近两年文献，引用总数达到 ≥20")

    # 7) 查重预估自检（深技大：知网重合率 <30% 合格、校级优秀 <20%）
    dup = check_duplication(md, cands)
    verdict["estimated_plagiarism_rate"] = dup["estimated_rate"]
    verdict["plagiarism_flag"] = dup["flag"]
    verdict["plagiarism_samples"] = dup["samples"]
    if dup["flag"]:
        _pct = f"{dup['estimated_rate'] * 100:.1f}%"
        _msg = ("估计重合率 " + _pct + "，已达或超过校级优秀线 20%（合格线 30%），"
                if dup["estimated_rate"] >= 0.20 else
                "估计重合率 " + _pct + "，接近校级优秀线 20%，")
        verdict["issues"].append("查重预估：" + _msg +
                                "请降低论文内部重复句与对参考文献标题的逐字复刻；"
                                "正式提交前仍须用知网做全网查重（本预估不含知网库）")
        verdict["suggestions"].append("改写重复/复刻段落，降低查重预估重合率；提交前用知网全网查重复核")

    # 综合与门控
    overall = round(sum(WEIGHTS[k] * verdict["scores"][k] for k in WEIGHTS), 1)
    verdict["overall"] = overall
    verdict["scores"]["overall"] = overall
    grade = "不及格"
    for name, thr in GRADE_BANDS:
        if overall >= thr:
            grade = name
            break
    verdict["grade"] = grade
    hard_ok = consistency == 100 and not hard
    verdict["gate"] = "hard_pass" if hard_ok else "hard_fail"
    verdict["passed"] = hard_ok and overall >= 60
    if not hard_ok:
        verdict["suggestions"].insert(0, "硬门未通过（一致性/谎言），必须修正后重做")
    elif overall < 60:
        verdict["suggestions"].insert(0, f"综合分 {overall} 低于及格线 60，需重做")
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
