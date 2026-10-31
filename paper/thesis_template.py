# -*- coding: utf-8 -*-
"""本科毕业设计（论文）结构模板 v1 —— AutoResearch 论文生成器的「单一事实来源」。

提炼自多所院校教务处 / 学院官方规范（湖北工业大学、吕梁学院、河南工学院等）、
计算机专业优秀毕业设计范文，并经反 AI 写作指南（AIGC 降痕、学术表达规范）校准。

后续迭代「让产出更像真实本科毕设」只改本文件：
  - 章节标题与编号（CHAPTERS）
  - 前置 / 后置必备部分（FRONT_MATTER / BACK_MATTER）
  - 文风规则（STYLE_RULES，注入大模型写作提示词，负责「反 AI 腔」）
  - 结构评分标尺（STRUCTURE_RUBRIC，供 EvalAgent 对照）
"""

# ── 1. 规范章节（实证 / 理工科型论文适用） ──────────────────────────
# key：generate_paper.py 内部逻辑名；no/cn/en：渲染出的章号、中文标题、英文标题。
# 内容来源与现有生成器完全对齐，仅改「章号 + 标题」，不改动确定性产物逻辑。
CHAPTERS = [
    {"key": "intro",       "no": 1, "cn": "绪论",            "en": "Introduction"},
    {"key": "related",     "no": 2, "cn": "国内外研究现状",  "en": "Related Work"},
    {"key": "method",      "no": 3, "cn": "研究方法",        "en": "Methodology"},
    {"key": "exp_setup",   "no": 4, "cn": "实验设置与数据",  "en": "Experimental Setup"},
    {"key": "results",     "no": 5, "cn": "实验结果与分析",  "en": "Results and Analysis"},
    {"key": "deep",        "no": 6, "cn": "深入分析与假设检验", "en": "In-Depth Analysis and Hypothesis Testing"},
    {"key": "discussion",  "no": 7, "cn": "讨论",            "en": "Discussion"},
    {"key": "conclusion",  "no": 8, "cn": "结论与展望",      "en": "Conclusion and Future Work"},
]
CHAPTER_BY_KEY = {c["key"]: c for c in CHAPTERS}


def chapter_heading(key: str) -> str:
    """给定逻辑名，返回「## 第N章 标题」形式的 Markdown 标题。"""
    c = CHAPTER_BY_KEY[key]
    return f"## 第{c['no']}章 {c['cn']}"


def chapter_toc_line(key: str) -> str:
    """目录里的一行：章号 + 中文 + 英文。"""
    c = CHAPTER_BY_KEY[key]
    return f"第{c['no']}章 {c['cn']}（{c['en']}）"


# ── 2. 前置 / 后置必备部分（非「章」，但本科毕设规范要求齐备） ────────
FRONT_MATTER = ["摘要（中文 + 英文）", "目录"]
BACK_MATTER = ["参考文献", "致谢", "附录：完整指标表"]


# ── 3. 文风规则（anti-AI，追加进 WRITER_SYSTEM 第 6 条） ───────────
# 源自反 AIGC 写作指南：去模板化、去机械连接词、长短句搭配、补全引用。
STYLE_RULES = """\
6. 文风须贴近真人撰写的本科毕业设计，消除机翻腔与 AI 痕迹：
   - 拒绝空泛套话（如「近年来……」「引起广泛关注」「具有重要意义」），改用具体数字、案例或研究视角切入。
   - 拒绝模糊引用（如「研究表明」「有学者认为」无出处），凡观点须补全 [编号] 与作者、年份。
   - 拒绝绝对化词汇（如「十分严重」「极其重要」），改用中性、证据驱动的表述。
   - 严禁用 bullet 列表堆砌论点；须融为逻辑连贯的段落（数据表与图注除外）。
   - 减少机械连接词（「首先/其次/最后」「综上所述」「值得注意的是」），靠句子内容的因果、递进、对比自然过渡，可交替使用「在此基础上」「从另一维度看」。
   - 段落须有明确主旨与逻辑链条，避免句子简单并列造成的逻辑跳跃。
   - 长短句搭配，刻意拉开句长差异，避免全文每句等长。
   - 适度引入低频但专业的表述，避免整段只围绕高频安全词汇打转。
   - 摘要用第三人称，300-600 字；中文摘要与英文摘要内容须一致。"""


# ── 4. 结构评分标尺（供 EvalAgent._score_structure 对照） ─────────
# 本科毕设必备章：绪论、国内外研究现状、研究方法、实验、结果、分析、结论、参考文献。
# 任一缺失即扣分；章号编号连续、含中英文摘要与目录视为加分项。
REQUIRED_CHAPTERS = ["绪论", "国内外研究现状", "研究方法", "实验设置与数据",
                     "实验结果与分析", "深入分析与假设检验", "讨论", "结论与展望"]
REQUIRED_BACK = ["参考文献"]


def structure_checklist() -> list[str]:
    """返回 EvalAgent 检查时必须出现的章节标题清单。"""
    return [chapter_heading(c["key"]) for c in CHAPTERS] + ["## 摘要", "## 目录", "## 参考文献"]
