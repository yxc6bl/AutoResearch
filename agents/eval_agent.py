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
from engine import llm_client as LLM  # noqa: E402

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
    "总而言之", "值得注意的是", "需要指出的是", "不难看出",
    "毋庸置疑", "近年来受到广泛关注", "随着", "在……背景下",
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


def check_innovation(md: str, faithful: bool = False):
    """创新维度。依据：哈工大创新与成效(25%)；北大⑤观点新颖；中科大创新与特色。

    结论章应明确创新点/独立见解/本文贡献，并区分本人与他人工作。
    faithful=True（忠实复现/对比选题，未提新方法）：放宽判定，认可「系统性对比/实证刻画」
    类贡献表述，不再因「无新算法」而判 0 分（那本就不是此类选题的要求）。
    """
    concl = _slice_chapter(md, "结论与展望", ["参考文献", "致谢"]) or md
    if faithful:
        _re = r'创新|独立见解|本文提出|本文贡献|本文设计|本文首次|新颖|本文系统比较|本文实证|本文刻画|本文工作|本文围绕|本文对比'
    else:
        _re = r'创新|独立见解|本文提出|本文贡献|本文设计|本文首次|新颖'
    has_innov = bool(re.search(_re, concl))
    has_section = bool(re.search(r'创新点|本文贡献|独立见解', md))
    issues = []
    if not has_innov:
        issues.append("结论未明确创新点 / 独立见解 / 本文工作贡献，或未区分本人与他人工作")
    elif not has_section:
        issues.append("结论提及贡献但缺少独立的「本文贡献」小节，建议单列")
    if faithful:
        score = 100 if has_innov else (70 if has_section else 40)
    else:
        score = 100 if (has_innov and has_section) else (60 if has_innov else 0)
    return max(0, score), issues


def check_writing_norms(md: str, mode: str = "general"):
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
    # 格式合规（按机构模式：general 通用下限 / sztu 深技大硬要求）并入撰写规范维度
    fmt_norm, fmt_norm_issues = check_format_compliance(md, mode)
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


def check_duplication(md: str, lit_candidates=None, mode: str = "general"):
    """本地查重预估（仅本地自检，不含知网全网库）。

    深技大规定：知网文字重合率 <30% 合格，校级优秀 <20%。本函数无法直接调
    知网，故用三路可控信号估算「论文自身重复 + 与参考文献标题逐字复刻」的重合率，
    作为提交前自检。信号：
      (1) 自重复：论文内两句话 4-gram Jaccard≥0.7 且长度接近 → 判定近重复；
      (2) 标题复刻：论文章节句与参考文献标题 4-gram Jaccard≥0.8 且重叠≥8字。
    返回 estimated_rate(0~1)、self_rate、title_rate、样本与告警级别。
    flag 仅在 mode="sztu" 时给出（查重为深技大硬性要求）；general 模式只做信息性预估，
    不触发不及格告警（各校本校查重口径不同，交由用户按所在学校复核）。
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
    if mode == "sztu":
        if est >= 0.30:
            flag = "fail(≥30% 合格线)"
        elif est >= 0.20:
            flag = "warn(≥20% 校级优秀线)"
    return {"estimated_rate": round(est, 4), "self_rate": round(self_rate, 4),
            "title_rate": round(title_rate, 4), "samples": samples, "flag": flag}


def check_format_compliance(md: str, mode: str = "general"):
    """格式合规（计入撰写规范维度），按机构模式收紧/放宽。

    general（默认，通用学术标准）：通用硬要求——关键词 3-4 个且每词≤6字、外文文献≥2 篇、
        中文摘要≥200 字、全文字数≥8000 字（通用下限，不作深技大 12000 硬线）、
        封面/诚信声明为推荐项（缺失仅轻微扣分）。
    sztu（深圳技术大学预设）：在通用要求之上叠加深技大公文通硬要求——全文字数≥12000、
        必备前置「封面」「诚信声明」（缺失重扣）。
    """
    issues = []
    score = 100
    sztu = (mode == "sztu")
    # 关键词
    m = re.search(r'关键词[：:]\s*(.+)', md)
    if m:
        kws = [k.strip().strip('*').strip('`').strip() for k in re.split(r'[；;]', m.group(1)) if k.strip()]
        if len(kws) > 4:
            issues.append(f"关键词 {len(kws)} 个（通用要求 3-4 个）")
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
            issues.append(f"中文摘要 {cn} 字（要求≥200字）")
            score -= 15
    # 全文字数
    cn_all = sum(1 for c in md if '一' <= c <= '鿿')
    word_floor = 12000 if sztu else 8000
    if cn_all < word_floor:
        issues.append(f"全文中文 {cn_all} 字（{'深技大要求' if sztu else '通用下限'}≥{word_floor}字）")
        score -= (15 if sztu else 10)
    # 必备前置
    if '诚信声明' not in md:
        issues.append("缺少「诚信声明」章节" + ("（深技大必备前置）" if sztu else "（建议补齐）"))
        score -= (10 if sztu else 5)
    if '封面' not in md:
        issues.append("缺少「封面」要素" + ("（深技大必备前置）" if sztu else "（建议补齐）"))
        score -= (10 if sztu else 5)
    # 外文文献≥2
    mr = re.search(r'#{1,4}\s*参考文献', md)
    ref = md[mr.start():] if mr else ""
    ref_items = re.findall(r'^\[\d+\].*$', ref, re.M)
    foreign = sum(1 for it in ref_items if re.search(r'[A-Za-z]{4,}', it))
    if foreign < 2:
        issues.append(f"外文文献 {foreign} 篇（要求≥2篇）")
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


def check_research_depth(md: str, analysis_json=None):
    """研究深度 / 论题复杂度评估（新方法，信息性维度，不计入加权分）。

    把论文归类到三种研究范式之一，并给出削弱论题深度的风险信号：
      - 原创/改进方法研究：明确提出新框架/算法/模型，并配消融或跨数据集验证。
      - 模型对比型实证研究：仅对现有方法做统一基准对比（本系统默认产出即此类）。
    同时检测「默认超参 / 小样本 / 缺消融」等削弱深度的信号，给出提升建议。
    """
    proposes = bool(re.search(
        r'本文提出[^\n。]{0,40}(框架|算法|模型|方法|网络|模块|策略|流程|机制)', md))
    default_hp = bool(re.search(
        r'默认超?参数|默认配置|默认设置|未进行.*调优|未系统.*调优|未对超参数|默认超参数配置', md))
    ablation = bool(re.search(r'消融|超参数.*搜索|网格搜索|参数寻优|调优实验|敏感性分析', md))
    multi = ("跨数据集" in md) or (
        isinstance(analysis_json, dict) and analysis_json.get("task_type") == "multi_dataset")
    n = None
    if isinstance(analysis_json, dict):
        n = (analysis_json.get("n_samples") or analysis_json.get("n")
             or analysis_json.get("sample_count") or analysis_json.get("num_samples"))
    small = (n is not None and isinstance(n, int) and n < 500)
    if proposes:
        level = "原创/改进方法研究" if (ablation or multi) else "改进型（有方法提出但缺消融/跨数据集验证）"
    else:
        level = "模型对比型实证研究"
    signals = {
        "proposes_method": proposes,
        "default_hyperparams": default_hp,
        "ablation_or_tuning": ablation,
        "multi_dataset": multi,
        "small_sample": small,
        "n_samples": n,
    }
    adv = []
    if not proposes:
        adv.append("未提出新方法，属现有方法对比型；若需提升论题深度，可在 spec 中增加「本文提出的改进方法」"
                   "（如融合集成 / 特征工程 / 超参数寻优）并配消融分析作为贡献点。")
    if default_hp:
        adv.append("全文使用默认超参数，方法潜力可能被低估；建议补充超参数寻优或敏感性分析。")
    if small:
        adv.append(f"样本量仅约 {n} 条，结论外推性受限；建议补充外部数据集或显式说明小样本适用性边界。")
    if not multi and not ablation:
        adv.append("缺消融 / 跨数据集验证，贡献的因果归因偏弱，易被判为「换数据集跑分」。")
    return {"level": level, "signals": signals, "advisory": adv}


def check_quality_profile(md: str, candidates=None, analysis_json=None, faithful: bool = False) -> dict:
    """整篇文章质量画像（新方法，信息性维度，不计入加权分）。

    直接回应「不要只判深度，整篇文章质量都要评判」：从六个互相独立的轴线给论文画像，
    每轴 0~100 并附诊断，再给综合分与最弱项。六轴：
      文献丰富度 / 数学严谨度 / 方法贡献度 / 实验严谨度 / 结构与叙事 / 写作规范。
    该画像只用于「量」与「诊断」，不参与 rubric 加权，故不会拉低既有总分。
    """
    prof: dict = {}
    cands = candidates or []
    n = len(cands)
    tiers = [c.get("venue_tier") or 0 for c in cands]
    top_ratio = (sum(1 for t in tiers if t >= 0.8) / n) if n else 0.0
    avg_q = (sum(c.get("quality_score", 0) or 0 for c in cands) / n) if n else 0.0
    lit = 100
    if n < 30:
        lit -= 30
    elif n < 45:
        lit -= 10
    if top_ratio < 0.2:
        lit -= 20
    elif top_ratio < 0.4:
        lit -= 10
    prof["literature_richness"] = {
        "score": max(0, lit), "count": n,
        "top_venue_ratio": round(top_ratio, 2), "avg_quality": round(avg_q, 2),
        "diagnosis": (f"入选文献 {n} 篇、顶刊/顶会占比 {top_ratio:.0%}；"
                      + ("文献量偏少或权威度不足，建议扩充至 40+ 篇并提升权威来源。"
                         if lit < 80 else "文献量与权威度达标。")),
    }
    # 数学严谨度
    eq_blocks = len(re.findall(r'式\s*[（(]\d+[）)]', md))
    has_deriv = bool(re.search(r'其中|推导|代入|展开|由式', md))
    math = 100
    if eq_blocks < 3:
        math -= 40
    elif eq_blocks < 5:
        math -= 20
    if not has_deriv:
        math -= 15
    prof["mathematical_rigor"] = {
        "score": max(0, math), "formula_count": eq_blocks, "has_derivation": has_deriv,
        "diagnosis": (f"编号公式 {eq_blocks} 个" + ("、缺符号推导说明" if not has_deriv else "、含符号推导")
                      + ("；公式密度偏低，建议补充问题形式化与评估框架。" if math < 80 else "。")),
    }
    # 方法贡献度（复用研究深度结论）
    d = check_research_depth(md, analysis_json)
    lvl = d["level"]
    mscore = {"原创/改进方法研究": 100,
              "改进型（有方法提出但缺消融/跨数据集验证）": 75,
              "模型对比型实证研究": 45}.get(lvl, 60)
    # 忠实复现模式：本文未提新方法本就是选题使然，按「严谨实证对比」给合理分，不判缺陷
    if faithful and lvl == "模型对比型实证研究":
        mscore = 70
        d = dict(d, level="忠实实证对比研究")
    prof["methodological_contribution"] = {
        "score": mscore, "level": lvl,
        "diagnosis": "方法贡献层级：" + lvl + ("；建议补充方法创新点。" if mscore < 80 else "；具备方法贡献。"),
    }
    # 实验严谨度
    exp = _slice_chapter(md, "实验设置与数据", ["实验结果与分析"]) or ""
    exp += _slice_chapter(md, "实验结果与分析", ["深入分析与"]) or ""
    exp += _slice_chapter(md, "深入分析与", ["讨论", "结论"]) or ""
    rig = {
        "cv": bool(re.search(r'交叉验证|随机种子|多种子|种子', exp or "")),
        "ablation": bool(re.search(r'消融|敏感性分析|参数寻优|网格搜索', exp or "")),
        "cross_dataset": ("跨数据集" in md),
        "significance": bool(re.search(r'显著性|mcnemar|卡方|χ²|chi|配对', (exp or "") + md)),
        "realnum": bool(re.search(r'\d+\.\d+%?|\d+%', md)),
    }
    es = 100
    if not rig["cv"]:
        es -= 25
    if not rig["significance"]:
        es -= 20
    if not rig["realnum"]:
        es -= 20
    if not (rig["ablation"] or rig["cross_dataset"]):
        es -= 15
    prof["experimental_rigor"] = {
        "score": max(0, es), **rig,
        "diagnosis": ("实验严谨性：含交叉验证/多种子" + ("与显著性检验" if rig["significance"] else "但缺显著性检验")
                      + ("；建议补充消融或跨数据集验证。" if not (rig["ablation"] or rig["cross_dataset"])
                         else "；含消融/跨数据集验证。")),
    }
    # 结构与叙事
    st, mc, mr = check_structure(md)
    narrative = bool(re.search(r'摘要|目录|本章小结|绪论', md))
    prof["structure_narrative"] = {
        "score": st, "missing_core": mc, "missing_rec": mr,
        "diagnosis": (("章节完整。" if not mc else f"缺核心章节：{mc}。")
                      + ("叙事要素齐全。" if narrative else "叙事要素不全。")),
    }
    # 写作规范
    wn, wn_is = check_writing_norms(md, "general")
    prof["writing_norms"] = {
        "score": wn, "issues": wn_is[:3],
        "diagnosis": f"撰写规范分 {wn}" + ("；存在套话/格式问题需修。" if wn < 80 else "；基本规范。"),
    }
    keys = ["literature_richness", "mathematical_rigor", "methodological_contribution",
            "experimental_rigor", "structure_narrative", "writing_norms"]
    overall = round(sum(prof[k]["score"] for k in keys) / len(keys))
    grade = "优秀" if overall >= 90 else ("良好" if overall >= 80 else ("中等" if overall >= 70 else "待提升"))
    prof["overall"] = overall
    prof["grade"] = grade
    prof["summary"] = (f"整篇文章质量画像综合 {overall} 分（{grade}）。"
                       f"最弱项见各分项诊断；优先补强最弱维度可最有效提升论题厚度与深度。")
    return prof


def _ground_truth(analysis: dict) -> str:
    """从真实实验结果压缩出「事实清单」，供 LLM 评审官比对论文是否编造数字。"""
    if not isinstance(analysis, dict):
        return "（无实验事实可对照）"
    bits = []
    bits.append(f"任务类型：{analysis.get('task_type','')}")
    if analysis.get("n_samples") is not None:
        bits.append(f"样本量：{analysis['n_samples']}")
    if analysis.get("n_features") is not None:
        bits.append(f"特征数：{analysis['n_features']}")
    bits.append(f"最优方法（按主指标）：{analysis.get('best_method','')} = {analysis.get('best_value','')}")
    metrics = analysis.get("metrics") or []
    for m in metrics[:12]:
        if isinstance(m, dict):
            bits.append(f"- {m.get('name','')} = {m.get('value','')}")
    if analysis.get("significance"):
        bits.append(f"显著性检验：{str(analysis['significance'])[:200]}")
    if analysis.get("task_type") == "association":
        sfs = analysis.get("single_fit_scores") or {}
        _rules = sfs.get("rules") or []
        if _rules:
            bits.append("关联规则（提升度降序，lift>1 为共现正相关；支持度/置信度/提升度均为实验实测，"
                        "论文引用时不得出现事实清单之外的数值）：")
            for _r in _rules[:12]:
                bits.append(f"  - {_r.get('rule')}：支持度={_r.get('support'):.4f}，"
                            f"置信度={_r.get('confidence'):.4f}，提升度={_r.get('lift'):.4f}")
    if analysis.get("seed_variance"):
        _sv_label = ("节点级各折准确率（折间波动，不是多种子方差）"
                     if analysis.get("graph") else "多种子方差")
        # 上限放宽到 2000：以前截到 200 字，只露出第一个方法的折值，
        # 评审官会把正文里其余真实折值判成「事实清单未提供」的编造。
        # 逐条列出并写明方法名，不能用 str(dict)[:N] 挤成一团：
        # 实测评审官曾把「随机森林_r2」的 std / seeds 当成「梯度提升树_r2」的，
        # 反过来把正文里完全正确的数字判成编造。结构化之后可按名字逐条核对。
        bits.append(f"{_sv_label}（每行都带方法名，请按方法名逐条核对，"
                    f"不要把相邻两行的数据串行）：")
        for _k, _v in analysis["seed_variance"].items():
            if isinstance(_v, dict):
                _seeds = "、".join(f"{float(x):.4f}"
                                  for x in (_v.get("seeds") or [])[:5])
                bits.append(f"  - {_k}：均值={float(_v.get('mean') or 0):.4f}，"
                            f"标准差={float(_v.get('std') or 0):.4f}，"
                            f"各次取值=[{_seeds}]")
            else:
                bits.append(f"  - {_k}：{_v}")
    _gr = analysis.get("graph")
    if isinstance(_gr, dict) and _gr:
        bits.append(f"图节点分类：{_gr.get('n_nodes')} 个节点、{_gr.get('n_edges')} 条边、"
                    f"{_gr.get('n_features')} 维特征、{_gr.get('n_classes')} 类，"
                    f"节点级分层 {_gr.get('n_splits')} 折交叉验证（不是随机种子复现、"
                    f"不是按固定比例划分训练集与测试集）。")
        bits.append("各方法×协议组合准确率（四位小数，论文引用时不得改写）：")
        for k, r in (_gr.get("results") or {}).items():
            bits.append(f"  - {k}：accuracy={r.get('accuracy'):.4f}，"
                        f"macro-F1={r.get('f1_macro'):.4f}，折间标准差={r.get('std'):.4f}")
        _b = _gr.get("best") or {}
        bits.append(f"核心结论（唯一最优组合）：{_b.get('key')} accuracy={_b.get('accuracy'):.4f}")
        bits.append("各方法×协议组合的逐折准确率（实测，正文引用时必须原样使用这些数值）：")
        for k, r in (_gr.get("results") or {}).items():
            bits.append(f"  - {k}：{r.get('fold_accuracy')}")
        bits.append("注意：配对差值由【未四舍五入】的逐折准确率计算，"
                    "与两个四位小数均值直接相减可能相差 0.0001；"
                    "论文必须原样引用 paired_tests 里的 diff，不得自行用均值相减推算差值。")
        bits.append(f"传导泄漏：{str(_gr.get('leakage_note'))[:240]}")
        bits.append(f"结论选取规则：{_gr.get('best_selection_note')}")
        bits.append("配对 t 检验（逐折配对，非 McNemar）：")
        for k, v in (analysis.get("paired_tests") or {}).items():
            bits.append(f"  - {k}：差值={v.get('diff'):.4f}，p={v.get('p'):.4f}")
        bits.append("本文是一跳邻居特征加表格分类器，不是图神经网络，未复现过平滑现象；"
                    "正文若声称使用 GNN/GCN、5 个随机种子、标准化/插补/独热编码，均为编造。")
    _im = analysis.get("image")
    if isinstance(_im, dict) and _im:
        bits.append(f"图像分类：{_im.get('dataset')}，官方划分 "
                    f"{_im.get('n_train_full')}/{_im.get('n_test_full')}，"
                    f"按种子 {_im.get('seed')} 分层抽取 "
                    f"{_im.get('n_train_used')}/{_im.get('n_test_used')}；"
                    f"输入形状 {_im.get('input_shape')}，{_im.get('n_classes')} 类；"
                    f"归一化={_im.get('normalization')}，增广={_im.get('augmentation')}，"
                    f"设备={_im.get('device')}，预训练权重={_im.get('pretrained')}。")
        bits.append(f"训练配置：epoch={_im.get('epochs')}，batch={_im.get('batch_size')}，"
                    f"随机种子={_im.get('seed')}，torch={_im.get('torch_version')}，"
                    f"合计耗时={_im.get('total_s')}s。")
        bits.append("各模型实测（四位小数，论文引用时不得改写）：")
        for k, v in (_im.get("models") or {}).items():
            extra = (f"，求解器={v.get('solver')}，最大迭代={v.get('max_iter')}"
                     if v.get("solver") else "")
            bits.append(f"  - {k}：accuracy={v.get('accuracy'):.4f}，"
                        f"macro-F1={v.get('f1_macro'):.4f}，参数量={v.get('params')}，"
                        f"epoch={v.get('epochs') or '不适用'}，训练耗时={v.get('train_s')}s{extra}")
        bits.append("逐 epoch 训练损失与测试准确率（实测，论文可原样引用）：")
        for k, hst in (_im.get("history") or {}).items():
            bits.append(f"  - {k}：" + "；".join(
                f"epoch{h['epoch']} 损失={h['train_loss']:.4f} 准确率={h['test_accuracy']:.4f}"
                for h in hst))
        _ib = _im.get("best") or {}
        bits.append(f"核心结论（最优模型）：{_ib.get('name')} "
                    f"accuracy={_ib.get('accuracy'):.4f}，macro-F1={_ib.get('f1_macro'):.4f}")
        bits.append("本实验固定划分、固定种子、单次训练，未做多种子复现与配对显著性检验；"
                    "正文若声称【本文】使用了多种子标准差、p 值、预训练权重、迁移学习、GPU、"
                    "数据增广、标准化/中位数插补/独热编码，均为编造（第 8.2 节作为"
                    "未来展望提及这些方向不算编造）。")
    # 逐类指标与混淆矩阵都是分析引擎实测的，必须完整进事实清单。
    # 早期版本只把它们塞进写作端的事实包、没给评审官，结果正文里每一个
    # 真实的逐类数字都被判成「事实清单未提供」的幻觉。
    if analysis.get("per_class"):
        bits.append("逐类指标（实测，论文可引用）：" +
                    json.dumps(analysis["per_class"], ensure_ascii=False)[:2000])
    if analysis.get("confusion_matrix"):
        bits.append("混淆矩阵（实测，论文可引用）：" +
                    json.dumps(analysis["confusion_matrix"], ensure_ascii=False)[:2000])
    if analysis.get("proposed_method"):
        bits.append(f"本文提出方法：{analysis['proposed_method']}（已做消融，消融结果见实验产物）")
    else:
        bits.append("本文未提出新方法（忠实复现/对比选题），不应声称提出新算法/框架")
    return "\n".join(bits)


def llm_judge(paper_path, analysis: dict, lit_count: int = 0, mode: str = "general") -> dict:
    """LLM 评审官：像答辩委员一样审论文，给六维分 + 找幻觉 + 给修订意见。

    这是「一个人写、另一个人审」里的「审」。它把论文与真实实验事实（ground truth）
    对照，专门抓编造数字 / 与裁决表矛盾 / 跑题 / 贡献夸大等 deterministic 规则难识别的问题。
    网关不可用时返回 {"available": False}，调用方回退到确定性 rules 评分。

    返回 dict：{
      available, scores(6维0-100), overall(0-100),
      hallucinations[...], issues[...], unsupported_claims[...], suggestions[...], verdict("pass"/"revise")
    }
    """
    if not LLM.available():
        return {"available": False, "reason": "gateway_unavailable"}
    try:
        md = Path(paper_path).read_text(encoding="utf-8")
    except Exception:
        return {"available": False, "reason": "paper_not_found"}

    gt = _ground_truth(analysis)
    prompt = (
        f"你是本科毕业设计（论文）答辩评委。请严格审阅下面这篇论文，并对照【真实实验事实】"
        f"判断论文是否编造数字、是否与事实矛盾、是否跑题、是否夸大贡献。\n\n"
        f"【真实实验事实（论文必须与之完全一致，不得出现事实清单之外的数字/方法/文献）】\n{gt}\n\n"
        f"【入选文献数量】{lit_count} 篇\n\n"
        f"【机构模式】{mode}（general=通用学术标准；sztu=深技大硬要求：≥12000字、封面/诚信声明必备）\n\n"
        "请输出 JSON：\n"
        "{\n"
        '  "scores": {"topic_sig":0-100, "literature":0-100, "structure":0-100,'
        ' "methodology":0-100, "innovation":0-100, "writing_norms":0-100},\n'
        '  "overall": 0-100 加权综合分,\n'
        '  "hallucinations": ["与事实清单矛盾/编造的具体处（如某数字、某方法名）"],\n'
        '  "issues": ["结构/逻辑/文献/深度方面的具体问题"],\n'
        '  "unsupported_claims": ["论文中无事实支撑、应改写或删除的断言（将反馈给写作端重写）"],\n'
        '  "suggestions": ["具体可执行的修改建议（每条都要落到章节或写法）"],\n'
        '  "verdict": "pass" 或 "revise"\n'
        "}\n"
        "评分要点：methodology 权重最高，重点看实验是否真实、对比基线/显著性/消融是否齐备；"
        "若论文出现事实清单之外的数字或方法，hallucinations 必须指出；"
        "若选题是忠实复现/对比（事实清单已注明未提新方法），不应因「无新算法」判不及格，"
        "而应看其对比深度与实证严谨度。\n"
        "判定口径（实测踩过坑，务必遵守）：\n"
        "1) 老师给的题目点名某方法（如「基于随机森林」）只指定研究对象，不要求它在对比中取得最优；"
        "实验显示别的方法更优而论文如实报告，不算编造也不算跑题。但论文不得声称点名方法是最优，"
        "也不得把甲方法的结果写成乙方法的结论。\n"
        "2) 四舍五入后的数值与事实清单高精度原值一致，不算编造；"
        "只有方法名、检验名、预处理操作、数据集、结论对象与事实清单矛盾，才算幻觉。\n"
        "3) 不要把「建议改进」写成幻觉；hallucinations 只放与事实清单明确矛盾的具体处。\n"
        "verdict 为 revise 当且仅当存在幻觉、严重结构缺失或核心事实矛盾。"
    )
    try:
        out = LLM.chat_json(prompt + "\n\n【论文全文】\n" + md,
                            system=LLM.FACT_FIRST_SYSTEM, temperature=0.2, max_tokens=2600)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"llm_error:{e}"}
    if not isinstance(out, dict) or "overall" not in out:
        return {"available": False, "reason": "bad_json"}
    sc = out.get("scores") or {}
    for k in WEIGHTS:
        try:
            sc[k] = float(sc.get(k, 0))
        except Exception:
            sc[k] = 0.0
    try:
        overall = float(out.get("overall", 0))
    except Exception:
        overall = sum(WEIGHTS[k] * sc.get(k, 0) for k in WEIGHTS)
    return {
        "available": True,
        "scores": sc,
        "overall": round(overall, 1),
        "hallucinations": out.get("hallucinations") or [],
        "issues": out.get("issues") or [],
        "unsupported_claims": out.get("unsupported_claims") or [],
        "suggestions": out.get("suggestions") or [],
        "verdict": out.get("verdict", "revise"),
    }


def run(paper_path, lit_candidates=None, store=None, out_path=None, mode: str = "general"):
    """主入口：对一篇论文给出评分 verdict。

    mode: "general"（默认，通用学术标准，985 rubric）或 "sztu"（深圳技术大学预设，
    叠加深技大公文通硬要求：12000 字、封面/诚信声明必备、查重硬门）。
    """
    md = Path(paper_path).read_text(encoding="utf-8")
    verdict = {"passed": False, "gate": "hard_fail", "scores": {}, "issues": [], "suggestions": []}

    # 预载实验事实（ground truth），供创新/质量画像判定「忠实复现 vs 方法贡献」。
    # 必须在 check_innovation（用到 _faithful）之前完成加载与派生，避免未赋值错误。
    _analysis_path = Path(paper_path).resolve().parent.parent / "03_experiment.json"
    _aj = None
    if _analysis_path.exists():
        try:
            _aj = json.loads(_analysis_path.read_text(encoding="utf-8"))
            if isinstance(_aj, dict) and isinstance(_aj.get("analysis"), dict):
                _aj = _aj["analysis"]
        except Exception:
            _aj = None
    # 忠实复现模式：本文未提新方法（proposed_method 为空）→ 放宽创新/贡献判定，
    # 避免把「忠实围绕题目方法做实证」误判为「无创新不及格」。
    _faithful = not (_aj and isinstance(_aj, dict) and _aj.get("proposed_method"))

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
    inn, inn_is = check_innovation(md, faithful=_faithful)
    verdict["scores"]["innovation"] = inn
    for i in inn_is:
        verdict["issues"].append("创新：" + i)
        verdict["suggestions"].append("结论明确创新点 / 独立见解，建议单列「本文创新点」小节")

    # 6) 撰写质量与学术规范
    wn, wn_is = check_writing_norms(md, mode)
    verdict["scores"]["writing_norms"] = wn
    for i in wn_is:
        verdict["issues"].append("撰写规范：" + i)
        if "套话" in i or "单句成段" in i or "排比" in i:
            verdict["suggestions"].append("降低套话与排比堆砌，提升反 AI 腔")
        if "图表" in i or "公式" in i:
            verdict["suggestions"].append("补充图表 / 公式编号引用，符合学术规范")
        if "参考文献" in i:
            verdict["suggestions"].append("补充近五年 / 近两年文献，引用总数达到 ≥20")

    # 7) 查重预估自检（深技大：知网重合率 <30% 合格、校级优秀 <20%；general 模式仅信息性预估）
    dup = check_duplication(md, cands, mode)
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

    # 8) 研究深度 / 论题复杂度（新维度，信息性，不计入加权分；直接回应「论题是否过简单」）
    depth = check_research_depth(md, _aj)
    verdict["research_depth"] = depth

    # 9) 整篇文章质量画像（六维，信息性，不计入加权分；直接回应「整篇质量如何评判」）
    profile = check_quality_profile(md, cands, _aj, faithful=_faithful)
    verdict["quality_profile"] = profile
    _qkeys = ("literature_richness", "mathematical_rigor", "methodological_contribution",
              "experimental_rigor", "structure_narrative", "writing_norms")
    _weak = min(_qkeys, key=lambda k: profile[k]["score"])
    verdict["suggestions"].append(
        f"整篇质量画像综合 {profile['overall']} 分（{profile['grade']}），最弱维度"
        f"「{_weak}」（{profile[_weak]['score']} 分）：{profile[_weak]['diagnosis']}")
    if depth["level"] == "模型对比型实证研究":
        verdict["suggestions"].insert(
            0,
            "本论文属「模型对比型」研究，与高质量范文（方法创新型）在方法原创性上存在本质差距；"
            "若需提升论题深度，可在 spec 中增加「本文提出的改进方法」并配消融/跨数据集验证。")
    for _a in depth["advisory"]:
        verdict["suggestions"].append("研究深度：" + _a)

    # 9) 范文对标（懒加载 benchmark_corpus，若存在且 corpus/ 有范文则对标）
    try:
        _bm_path = ROOT / "benchmark_corpus.py"
        _corpus_dir = ROOT / "corpus"
        if _bm_path.exists() and _corpus_dir.exists():
            import importlib.util as _iu
            _spec = _iu.spec_from_file_location("benchmark_corpus", str(_bm_path))
            _bm = _iu.module_from_spec(_spec)
            _spec.loader.exec_module(_bm)
            _rep = _bm.benchmark(str(paper_path), str(_corpus_dir))
            verdict["corpus_benchmark"] = _rep
            _gap = _rep.get("headline_gap")
            if _gap:
                verdict["suggestions"].append("范文对标：" + _gap)
    except Exception:
        pass

    # 10) LLM 评审官（「第二双眼睛」）：像答辩委员一样审，抓幻觉/跑题/夸大。
    #     确定性硬门（一致性/谎言）仍是安全网不可绕过；此处把 LLM 评判作为补充信号，
    #     若存在幻觉或 verdict=revise 则强制不通过（人审也会这么判）。
    _lj_bad = False
    try:
        _lj = llm_judge(str(paper_path), _aj, len(cands) if cands else 0, mode)
        # 评审官缺席也必须留痕：以前 available=False 时 verdict 里连字段都没有，
        # 下游分不清「审过没发现幻觉」和「根本没审」，会把未验证当成通过。
        verdict["llm_judge"] = _lj if isinstance(_lj, dict) else {
            "available": False, "reason": "bad_result"}
        if _lj.get("available"):
            verdict["llm_judge"] = _lj
            _lj_bad = bool(_lj.get("hallucinations"))
            if _lj_bad:
                verdict["issues"].insert(0, "LLM评审官发现疑似幻觉：" + "；".join(_lj["hallucinations"][:5]))
                verdict["suggestions"].insert(0, "按 LLM 评审官意见回滚改写，消除虚构数字/事实")
                verdict["gate"] = "hard_fail"
                verdict["passed"] = False
            elif _lj.get("verdict") == "revise":
                verdict["suggestions"].append("LLM评审官建议修订：" + "；".join(_lj.get("suggestions", [])[:3]))
    except Exception as e:  # noqa: BLE001
        verdict["llm_judge"] = {"available": False, "reason": f"exception:{e}"}
        print(f"[EvalAgent] LLM 评审官调用失败（不影响规则评分）：{e}")

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
    # 幻觉拦截必须参与最终裁决。此前这里只用确定性一致性重算 hard_ok，
    # 把上面第 10 步刚设的 gate=hard_fail / passed=False 又覆盖回 hard_pass，
    # 导致「评审官查出幻觉、评分门却给 99 分优秀并通过」——实测已复现。
    hard_ok = consistency == 100 and not hard and not _lj_bad
    verdict["llm_hallucination_gate"] = bool(_lj_bad)
    verdict["gate"] = "hard_pass" if hard_ok else "hard_fail"
    verdict["passed"] = hard_ok and overall >= 60
    if not hard_ok:
        verdict["suggestions"].insert(0, "硬门未通过（一致性/谎言/评审官幻觉），必须修正后重做")
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
    ap.add_argument("--mode", default="general",
                    help="机构模式：general（默认，通用学术标准）或 sztu（深圳技术大学预设）")
    args = ap.parse_args()
    cands = None
    if args.lit and Path(args.lit).exists():
        try:
            d = json.loads(Path(args.lit).read_text(encoding="utf-8"))
            cands = d if isinstance(d, list) else (d.get("papers") or d.get("candidates") or [])
        except Exception:
            cands = None
    v = run(args.paper, lit_candidates=cands, out_path=args.out, mode=args.mode)
    print(json.dumps(v, ensure_ascii=False, indent=2))
    sys.exit(0 if v["passed"] else 1)


if __name__ == "__main__":
    main()
