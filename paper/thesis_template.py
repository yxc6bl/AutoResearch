# -*- coding: utf-8 -*-
"""本科毕业设计（论文）结构模板 v2 —— AutoResearch 论文生成器的「单一事实来源」。

v2 来源升级为 985/211 级别，而非普通院校格式规范：
  - 清华大学计算机专业毕业论文规范（八大部分结构）
  - 北京大学工学院 / 信息科学技术学院本科毕业论文要求与「优秀论文评定标准」
    （明确要求：独立查阅文献能力、基本理论扎实、分析解决问题能力、有新意与独立见解、写作规范逻辑性强）
  - 上海交通大学优异学士学位论文（TOP 1%）评选标准与抽检评议要素
    （专业能力 / 分析能力 / 研究新意 / 引用规范 / 理工科正文字数≥2万、复制比<10%）
  - 公开可获取的清北 ML 方向本科毕设范文写作特征（域迁移与场景重建、可信机器学习、
    基于元学习的半监督算法、眼动数据进食障碍预测等），其导师评语反复强调：
    「结构清晰、写作规范、行文准确、逻辑严谨、实验数据充分、有新意、独立见解」
  - 并经反 AIGC 学术写作指南校准（去 AI 痕迹、学术表达规范）

后续迭代「让产出更像真实 985 本科毕设」只改本文件：
  - 章节标题与编号（CHAPTERS）
  - 前置 / 后置必备部分（FRONT_MATTER / BACK_MATTER）
  - 文风规则（STYLE_RULES，注入大模型写作提示词，负责「反 AI 腔 + 内容严谨度」）
  - 结构评分标尺（structure_checklist）与内容质量标尺（content_quality_checklist，供 EvalAgent）
"""

# ── 1. 规范章节（实证 / 理工科型论文「默认 8 章」骨架） ──────────────────
# 8 章是「实证 / 理工科型」默认骨架（绪论→研究现状→方法→实验→结果→深入分析→讨论→结论）。
# 真实 985 毕设章数按研究类型浮动：研究型（多项主要工作）常拆「方法章」（如北大 2024 为 5 章：
# 引言 + 国内外研究现状 + 两项方法 + 总结与展望）；但「前置（中英文摘要 + 目录）+ 后置（参考文献 + 致谢 + 附录）
# 齐备、且每章带『本章小结』」是跨校共性硬要求。generate_paper 按本骨架渲染，研究型可在 planner 阶段
# 调整章数与标题；模板只定「不变项」，升级只加法。
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


# ── 3. 文风规则（anti-AI + 985 内容严谨度，追加进 WRITER_SYSTEM 第 6 条） ──
# 反 AI 部分源自反 AIGC 写作指南；内容严谨度部分提炼自清北 / 上交优秀论文标准与范文特征。
STYLE_RULES = """\
6. 文风须贴近真人撰写的 985 高校本科毕业设计，消除机翻腔、AI 痕迹，并满足内容严谨度要求：

   【反 AI 腔】
   - 拒绝空泛套话（如「近年来……」「引起广泛关注」「具有重要意义」），改用具体数字、案例或研究视角切入。
   - 拒绝模糊引用（如「研究表明」「有学者认为」无出处），凡观点须补全 [编号] 与作者、年份。
   - 拒绝绝对化词汇（如「十分严重」「极其重要」），改用中性、证据驱动的表述。
   - 严禁用 bullet 列表堆砌论点；须融为逻辑连贯的段落（数据表与图注除外）。
   - 减少机械连接词（「首先/其次/最后」「综上所述」「值得注意的是」），靠句子内容的因果、递进、对比自然过渡，可交替使用「在此基础上」「从另一维度看」。
   - 段落须有明确主旨与逻辑链条，避免句子简单并列造成的逻辑跳跃。
   - 长短句搭配，刻意拉开句长差异，避免全文每句等长。
   - 适度引入低频但专业的表述，避免整段只围绕高频安全词汇打转。
   - 摘要用第三人称，300-600 字；中文摘要与英文摘要内容须一致。

   【985 内容严谨度】
   - 文献综述（第2章）须按「主题 / 待解决问题」组织成逻辑故事，而非逐篇罗列流水账；阐明各前人工作的立场与局限，并指出本文与它们的关系与差异。必须出现 [编号] 引用。
   - 每章正文末尾设一段「本章小结」，独立归纳本章核心工作与结论，不与正文重复。
   - 结论章（第8章）须明确、分条列出本文的创新点 / 独立见解，并严格区分「本人工作」与「他人成果」；同时给出具体的研究局限与可落地的未来展望。
   - 行文一律客观第三人称（「本文」「本研究」），杜绝口语化（「我觉得」「众所周知」「非常」「特别」）与主观副词。
   - 实验章节须说明对比基线、评价指标与稳健性证据（多随机种子方差 / 交叉验证）或消融分析；图表应脱离正文即可被理解（独立编号、采用三线表）。
   - 图表遵循「先编号后引用」（如「如图 3-1 所示」）；公式独立居中编号并解释各符号含义。
   - 参考文献力求 ≥20 篇，近五年文献占比 ≥1/3，并包含近两年代表性文献；引用资料应具有权威性与直接参考价值。

   【真实 985 范文写法示范】（提炼自北大 2024 计算机毕设、清华 2008 计算机毕设，仅提取「写法特征」，不抄内容）
   - 摘要：以概念定义或具体问题切入，杜绝「近年来……引起广泛关注」式套话；用「本文聚焦于……」明确任务；创新点用「针对 X 问题，本文提出……」分条陈述，每条含方法 + 结果。
   - 文献综述：按「主题 / 待解决问题」组织，每条前人工作须点明其立场与局限，并以「然而……」「其不足在于……」引出本文差异；严禁逐篇流水账罗列。
   - 本章小结：归纳本章核心工作，指出相关局限，并铺垫后文（如「为后文……所做的改进工作进行铺垫」），不与正文重复。
   - 总结章：分条列出创新点 / 独立见解，每条「针对 X 问题，本文提出 Y，在 Z 上展现出……」；未来展望须基于本文具体局限给出可落地方向，避免空话。
   - 引用：正文用 [编号] 上标式标注；参考文献含作者、标题、出处、年份，近五年文献占相当比例。
   - 章号：本科毕设普遍用「第 N 章」（阿拉伯）或「第一章」（中文），按所在学校模板选定；节号用「1.1」式。"""


# ── 4. 结构评分标尺（供 EvalAgent._score_structure 对照） ─────────
# 本科毕设必备章：绪论、国内外研究现状、研究方法、实验、结果、分析、结论、参考文献。
# 任一缺失即扣分；章号编号连续、含中英文摘要与目录视为加分项。
REQUIRED_CHAPTERS = ["绪论", "国内外研究现状", "研究方法", "实验设置与数据",
                     "实验结果与分析", "深入分析与假设检验", "讨论", "结论与展望"]
REQUIRED_BACK = ["参考文献"]


def structure_checklist() -> list[str]:
    """返回 EvalAgent 检查时必须出现的章节标题清单。"""
    return [chapter_heading(c["key"]) for c in CHAPTERS] + ["## 摘要", "## 目录", "## 参考文献"]


# ── 5. 内容质量标尺（供 EvalAgent._score_content_quality 对照，985 级门槛） ──
# 这是 985 与普通毕设的真正差距所在：不在于格式，而在于论证深度与严谨度。
CONTENT_QUALITY_RULES = [
    {
        "name": "文献综述逻辑组织",
        "pass_hint": "第2章按主题/待解决问题组织，每条前人工作点明立场与局限，并以「然而…」引出本文差异，含 [编号] 引用",
        "fail_hint": "文献综述逐篇流水账罗列，缺主题组织、缺前人局限与本文差异定位",
    },
    {
        "name": "本章小结",
        "pass_hint": "各章正文末尾含「本章小结」，归纳核心工作 + 指出局限 + 铺垫后文，不与正文重复",
        "fail_hint": "缺少各章小结，或小结仅重复正文、无归纳收束",
    },
    {
        "name": "创新点 / 独立见解明确",
        "pass_hint": "结论章（默认第8章，按实际章号）分条列出创新点 / 独立见解（「针对 X 问题，本文提出 Y」），并区分本人与他人工作",
        "fail_hint": "结论未分条明确创新点，或未区分本人工作与他人成果",
    },
    {
        "name": "实验严谨度",
        "pass_hint": "含对比基线、评价指标与稳健性（方差 / 交叉验证）或消融分析",
        "fail_hint": "实验叙述单薄，缺对比基线或稳健性证据",
    },
    {
        "name": "图表与公式规范",
        "pass_hint": "图表先编号后引用、可脱离正文理解；公式独立编号",
        "fail_hint": "图表编号 / 引用不规范，或图表无法脱离正文理解",
    },
    {
        "name": "参考文献质量与时效",
        "pass_hint": "参考文献 ≥20 篇且含近五年与近两年文献",
        "fail_hint": "参考文献偏少或缺乏近五年 / 近两年文献",
    },
]


def content_quality_checklist() -> list[dict]:
    """返回 EvalAgent 评估内容严谨度时逐条核查的规则。"""
    return CONTENT_QUALITY_RULES


# ── 6. 论文类型注册表（可插拔骨架，v3 新增） ──────────────────────────
# 系统此前只会写「方法对比型实证研究」一种骨架（CHAPTERS）。本注册表把「章节骨架」
# 抽成可按 paper_type 调度的配置，使同一套引擎能覆盖更多论文类型：
#   - method_comparison：既有默认骨架（需跑实验，产生指标/对比表）
#   - survey           ：文献综述型（纯文献，不做实验，按主题系统梳理前人工作）
#   - system_design    ：系统/软件设计型（纯 spec 驱动，讲需求/架构/模块/验证）
# 新增类型只需在此登记一套 chapters，并在 generate_paper.py 写对应的 builder 即可。
PAPER_TYPES = {
    "method_comparison": {
        "label": "方法对比型实证研究",
        "requires_experiment": True,
        "default_task_type": "classification",
        "chapters": CHAPTERS,
    },
    "survey": {
        "label": "文献综述型",
        "requires_experiment": False,
        "default_task_type": "survey",
        "chapters": [
            {"key": "intro",        "no": 1, "cn": "绪论",               "en": "Introduction"},
            {"key": "method_survey","no": 2, "cn": "文献检索与分析方法", "en": "Literature Retrieval and Analysis Method"},
            {"key": "framework",    "no": 3, "cn": "研究主题分类框架",   "en": "Taxonomy of Research Topics"},
            {"key": "status",       "no": 4, "cn": "各主题研究现状",     "en": "State of the Art by Topic"},
            {"key": "trend",        "no": 5, "cn": "研究趋势与争论",     "en": "Trends and Debates"},
            {"key": "open",         "no": 6, "cn": "开放问题与未来方向", "en": "Open Problems and Future Directions"},
            {"key": "conclusion",   "no": 7, "cn": "结论与展望",         "en": "Conclusion and Future Work"},
        ],
    },
    "system_design": {
        "label": "系统/软件设计型",
        "requires_experiment": False,
        "default_task_type": "system_design",
        "chapters": [
            {"key": "intro",       "no": 1, "cn": "绪论",           "en": "Introduction"},
            {"key": "requirements", "no": 2, "cn": "需求分析",       "en": "Requirements Analysis"},
            {"key": "architecture", "no": 3, "cn": "总体架构设计",   "en": "System Architecture"},
            {"key": "modules",      "no": 4, "cn": "核心模块设计与实现", "en": "Core Module Design and Implementation"},
            {"key": "tech",         "no": 5, "cn": "关键技术与难点", "en": "Key Techniques and Challenges"},
            {"key": "validation",   "no": 6, "cn": "系统验证与测试", "en": "System Validation and Testing"},
            {"key": "conclusion",   "no": 7, "cn": "结论与展望",     "en": "Conclusion and Future Work"},
        ],
    },
    "case_study": {
        "label": "案例研究型",
        "requires_experiment": False,
        "default_task_type": "case_study",
        "chapters": [
            {"key": "intro",       "no": 1, "cn": "绪论",               "en": "Introduction"},
            {"key": "theory",      "no": 2, "cn": "理论基础与文献回顾", "en": "Theoretical Basis and Literature Review"},
            {"key": "design",      "no": 3, "cn": "研究设计与案例选择", "en": "Research Design and Case Selection"},
            {"key": "context",     "no": 4, "cn": "案例背景与现状",     "en": "Case Background and Status Quo"},
            {"key": "analysis",    "no": 5, "cn": "案例分析",           "en": "Case Analysis"},
            {"key": "discussion",  "no": 6, "cn": "讨论与启示",         "en": "Discussion and Implications"},
            {"key": "conclusion",  "no": 7, "cn": "结论与展望",         "en": "Conclusion and Future Work"},
        ],
    },
    "technical_report": {
        "label": "工程实践报告型",
        "requires_experiment": False,
        "default_task_type": "technical_report",
        "chapters": [
            {"key": "intro",          "no": 1, "cn": "概述",       "en": "Overview"},
            {"key": "requirements",   "no": 2, "cn": "需求与约束", "en": "Requirements and Constraints"},
            {"key": "architecture",   "no": 3, "cn": "方案设计",   "en": "Solution Design"},
            {"key": "implementation", "no": 4, "cn": "实施过程",   "en": "Implementation Process"},
            {"key": "results",        "no": 5, "cn": "结果与验证", "en": "Results and Verification"},
            {"key": "issues",         "no": 6, "cn": "问题与改进", "en": "Problems and Improvements"},
            {"key": "conclusion",     "no": 7, "cn": "总结与展望", "en": "Summary and Outlook"},
        ],
    },
    "research_proposal": {
        "label": "研究方案设计型（开题）",
        "requires_experiment": False,
        "default_task_type": "research_proposal",
        "chapters": [
            {"key": "intro",       "no": 1, "cn": "研究背景与意义",   "en": "Background and Significance"},
            {"key": "status",      "no": 2, "cn": "国内外研究现状",   "en": "State of the Art"},
            {"key": "questions",   "no": 3, "cn": "研究问题与目标",   "en": "Research Questions and Objectives"},
            {"key": "content",     "no": 4, "cn": "研究内容与方法",   "en": "Research Content and Methods"},
            {"key": "route",       "no": 5, "cn": "技术路线与可行性", "en": "Technical Route and Feasibility"},
            {"key": "plan",        "no": 6, "cn": "进度安排与预期成果", "en": "Schedule and Expected Outcomes"},
            {"key": "conclusion",  "no": 7, "cn": "结语",             "en": "Conclusion"},
        ],
    },
}


# 非实验型（只靠「素材 → 结构化成文」）的论文类型集合。
NON_EXPERIMENTAL_TYPES = tuple(
    k for k, v in PAPER_TYPES.items() if not v.get("requires_experiment", True)
)


def paper_type_registry() -> dict:
    """返回论文类型注册表（键 → {label, requires_experiment, default_task_type, chapters}）。"""
    return PAPER_TYPES


def is_experimental(paper_type: str) -> bool:
    """该论文类型是否需要跑确定性实验（产生 03_experiment.json 的指标）。"""
    return PAPER_TYPES.get(paper_type, PAPER_TYPES["method_comparison"]).get("requires_experiment", True)


def chapters_for(paper_type: str) -> list[dict]:
    """返回某论文类型的章节骨架（与 CHAPTERS 同构的 dict 列表）。"""
    return PAPER_TYPES.get(paper_type, PAPER_TYPES["method_comparison"])["chapters"]


def chapter_heading_of(ch: dict) -> str:
    """给定章节 dict（含 no/cn），返回「## 第N章 标题」形式的 Markdown 标题。"""
    return f"## 第{ch['no']}章 {ch['cn']}"


def heading_to_key(chapters: list[dict]) -> dict:
    """构造 heading→key 映射，供章节小结插入逻辑使用。"""
    return {chapter_heading_of(c): c["key"] for c in chapters}


def structure_checklist_for(paper_type: str) -> list[str]:
    """返回 EvalAgent 结构检查时必须出现的章节标题清单（按类型不同）。"""
    chs = chapters_for(paper_type)
    lines = ["## 摘要", "## 目录"]
    for c in chs:
        lines.append(chapter_heading_of(c))
    lines.append("## 参考文献")
    return lines
