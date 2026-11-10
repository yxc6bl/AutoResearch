"""非实验型论文的「类型规格」——把「素材 → 分章提示 → 兜底文案」声明成数据。

设计目标（可扩展性）
--------------------
新增一种论文类型，只做两件事：

1. 在 ``paper/thesis_template.py`` 的 ``PAPER_TYPES`` 里登记 ``chapters``（章节骨架）；
2. 在本文件 ``TYPE_SPECS`` 里登记一份规格：素材来源、分批写作提示、兜底文案、附加小节。

**不需要**再写新的 builder 函数——``paper/generate_paper.py`` 的
``_build_non_experimental`` 会读取本文件的规格，统一完成「读素材 → 调 LLM →
字段级兜底 → 拼装 → 收尾」全流程。

零编造约定
----------
每种类型只允许使用自己声明的素材：
- ``literature``     ：01_literature.json（题名/作者/来源/年份/链接，**无摘要无指标**）
- ``system_spec``    ：task_spec.json 的 system_spec
- ``case_spec``      ：task_spec.json 的 case_spec
- ``project_spec``   ：task_spec.json 的 project_spec
- ``proposal_spec``  ：task_spec.json 的 proposal_spec

模型只能在这些素材范围内写，不得引入素材外的数字、技术、实验结论。
"""
from __future__ import annotations

import re

# ── 通用零编造条款 ─────────────────────────────────────────────────────
_NO_FABRICATE_LITERATURE_PLAIN = (
    "## ⚠️ 零编造硬约束（违背即判不可交付）\n"
    "你只拥有每篇文献的【题名 / 作者 / 来源(venue) / 年份 / 链接】，"
    "**没有摘要、没有正文、没有任何具体指标数字**。因此：\n"
    "1. 正文只能**定性**描述各文献的研究问题、方法路线类别、应用场景，"
    "以及『从题名可合理推断的贡献方向』；\n"
    "2. **严禁编造任何具体性能数字**（准确率、AUC、F1、敏感度、C-index、Dice 等），"
    "严禁编造实验设置、数据集规模、样本量；\n"
    "3. 若某文献题名已点明方法（如 'Deep learning for ...'、'A survey of ...'），"
    "可据此定性转述其方法类别，但**不得补造数字或结论**；\n"
    "4. 所有陈述必须能在下列『参考文献清单』的题名/来源/年份中找到依据或可合理推断，否则删除；\n"
    "5. 这类论文的价值在于『脉络、分歧、开放问题』的归纳，不在于列数字。"
)

_NO_FABRICATE_LITERATURE_WITH_ABSTRACTS = (
    "## ⚠️ 零编造硬约束（违背即判不可交付）\n"
    "你拥有每篇文献的【题名 / 作者 / 来源 / 年份 / 链接】，以及**部分文献的真实摘要节选**"
    "（见下方「文献摘要节选」，带 [编号]）。摘要里的具体数值已脱敏为「〔数值略〕」。\n"
    "1. **有摘要的文献**：可转述该摘要明确写出的研究问题、方法路线、数据来源与结论方向，"
    "须用「该文献报告……」「[N] 该研究……」等带归属的句式；\n"
    "2. **全文严禁任何具体数字**（样本量、记录数、准确率、AUC、F1、耗时、并发数等）；"
    "摘要已脱敏，你看不到这些数字，写了即视为编造。年份与参考文献编号除外；\n"
    "3. **无摘要的文献**：只能从题名定性推断其方法类别与应用方向，"
    "严禁补造摘要、结论、数据集或数字；\n"
    "4. 严禁把文献 A 的方法、数据或结论安到文献 B 或本文头上；"
    "严禁把任何文献的结果写成「本文实验得到」「本系统实测达到」；严禁编造清单之外的文献；\n"
    "5. 所有陈述必须能在题名或**对应编号**的摘要节选中找到依据，否则删除；\n"
    "6. 这类论文的价值在于『脉络、分歧、开放问题』的归纳，不在于堆砌数字。"
)


def _no_fabricate_literature(has_abstracts: bool) -> str:
    """零摘要的旧文献池保持原行为，避免历史产物回归。"""
    return (_NO_FABRICATE_LITERATURE_WITH_ABSTRACTS if has_abstracts
            else _NO_FABRICATE_LITERATURE_PLAIN)

_NO_FABRICATE_SPEC = (
    "## ⚠️ 零编造硬约束（违背即判不可交付）\n"
    "1. 正文只能基于下面给出的素材展开，**不得引入素材外的技术、组件、模块或数字**；\n"
    "2. **严禁编造实测/实验得到的结果数字**（响应时延、准确率、并发数、样本量、采样率等）。"
    "素材里给的是『目标值/验收标准』时，只能写『设计目标为…』『验收标准为…』，"
    "不得写成『实测达到…』；\n"
    "3. 模块职责、接口、数据流、类别标签一律以素材描述为准，不得虚构未声明的功能，"
    "也不得增删类别；\n"
    "4. 若某项信息素材未给出，正确做法是**不写**，而不是补一个看起来合理的值；\n"
    "5. **全文任何位置的数字都必须能在素材中逐字找到依据**（含验收阈值、偏差百分比、"
    "耗时分钟数、人数、次数、版本号）。落笔前逐句自检：这个数字素材里有吗？没有就删掉，"
    "改成『以验收时实测为准』『按试点结果确定』这类不含数字的表述。"
)


# ── 素材渲染 ──────────────────────────────────────────────────────────
def render_block(block, prefix: str = "") -> list[str]:
    """把任意 dict / list 结构渲染成「键：值」事实行（供提示词使用）。"""
    out: list[str] = []
    if isinstance(block, dict):
        for k, v in block.items():
            key = f"{prefix}{k}"
            if isinstance(v, str):
                if v.strip():
                    out.append(f"{key}：{v.strip()}")
            elif isinstance(v, bool):
                out.append(f"{key}：{'是' if v else '否'}")
            elif isinstance(v, (int, float)):
                out.append(f"{key}：{v}")
            elif isinstance(v, list):
                if not v:
                    continue
                if all(isinstance(x, str) for x in v):
                    out.append(f"{key}：" + "；".join(v))
                else:
                    for i, item in enumerate(v, 1):
                        if isinstance(item, dict):
                            name = item.get("name") or item.get("title") or f"#{i}"
                            pairs = [f"{ik}={iv}" for ik, iv in item.items()
                                     if ik not in ("name", "title")
                                     and isinstance(iv, (str, int, float)) and str(iv).strip()]
                            out.append(f"{key}[{name}]：" + "；".join(pairs))
                        else:
                            out.append(f"{key}[{i}]：{item}")
            elif isinstance(v, dict):
                out.extend(render_block(v, prefix=f"{key}."))
    elif isinstance(block, list):
        for i, item in enumerate(block, 1):
            if isinstance(item, str):
                out.append(f"{i}. {item}")
            elif isinstance(item, dict):
                out.extend(render_block(item, prefix=f"{prefix}{i}."))
    elif isinstance(block, str) and block.strip():
        out.append(block.strip())
    return out


def _spec_material(paper_type: str, spec: dict, block_key: str, label: str,
                   extra_rules: list[str] | None = None) -> str:
    """spec 类素材的统一事实包。"""
    block = spec.get(block_key) or spec.get(block_key.replace("_spec", "")) or {}
    lines = [f"# {label}事实素材（零编造：以下内容均来自 task_spec.json 的 {block_key} 字段）",
             f"论文主题：{_topic_of(spec)}",
             "",
             _NO_FABRICATE_SPEC]
    if extra_rules:
        lines.extend(extra_rules)
    lines.append("")
    if block:
        lines.append(f"## {label}素材（{block_key}）")
        lines.extend(render_block(block))
    else:
        lines.append(f"## ⚠️ 未提供 {block_key}")
        lines.append(f"task_spec.json 中没有 {block_key} 字段。请只依据题目与常识范围做"
                     f"**通用性**论述，并在文中明确说明本方案为基于题目推演的设计，"
                     f"不得虚构具体的产品参数、实测数据或未声明的功能。")
    return "\n".join(lines)


def _literature_material(spec: dict, refs: list, references_block: str,
                         abstract_text: str = "") -> str:
    topic = _topic_of(spec)
    lines = ["# 文献事实素材（零编造：以下内容均来自 01_literature.json，禁止引入素材外的文献/数字）",
             f"论文主题：{topic}",
             f"入选文献总数：{len(refs)} 篇",
             "",
             _no_fabricate_literature(bool(abstract_text))]
    by_role: dict[str, int] = {}
    for r in refs:
        role = (r.get("role") or "未分类")
        by_role[role] = by_role.get(role, 0) + 1
    if by_role:
        lines.append("文献角色分布：" + "；".join(f"{k} {v} 篇" for k, v in by_role.items()))
    years = [int(r["year"]) for r in refs
             if str(r.get("year", "")).strip().isdigit()]
    if years:
        near = sum(1 for y in years if y >= 2021)
        lines.append(f"文献年份跨度：{min(years)}–{max(years)}（近五年 {near} 篇）")
    lines += ["",
              f"## ⚠️ 引用编号范围：严格限定 [1]–[{len(refs)}]",
              f"正文只允许出现 [1] 到 [{len(refs)}] 的编号；出现 [{len(refs) + 1}] 及以上的编号"
              "即视为编造文献（不得凭印象续编号，也不得写「[31] 至 [40]」这类超出范围的区间）。",
              "",
              "## 参考文献清单（正文须用 [编号] 引用，不得编造素材外文献）",
              references_block or "（无文献数据）"]
    return "\n".join(lines)


def _topic_of(spec: dict) -> str:
    return (spec.get("title") or spec.get("topic") or spec.get("direction")
            or "（未提供主题）").strip()


def build_fact_pack(paper_type: str, spec: dict, refs: list,
                    references_block: str, lit: dict | None = None) -> tuple[str, dict]:
    """按类型返回 (事实包提示词前缀, 摘要统计)。

    ``lit`` 传入 01_literature.json 的内容时，会把该文献池里**真实存在**的摘要
    节选接进事实包。写作端与 ``eval_agent._ground_truth_for`` 共用
    ``paper.lit_abstracts.abstract_block``，保证两边逐字同源。
    """
    from paper.lit_abstracts import abstract_block, enabled
    from paper.lit_material import gist_block
    papers = ((lit or {}).get("papers") or []) if isinstance(lit, dict) else []
    ab_text, ab_stats = (abstract_block(papers) if enabled()
                         else ("", {"n_total": len(papers), "n_with_abstract": 0,
                                    "n_used": 0, "chars": 0}))
    # 一句话概要是检索阶段落盘的真实素材，一直只出现在正文引用块里、没进写作提示，
    # 模型写「研究现状」时只看得到题名，于是凭题名脑补文献结论。这里补上。
    gist_text, gist_stats = gist_block(papers, audience="writer")
    stats = {**ab_stats, "gists_used": gist_stats["n_used"]}

    s = TYPE_SPECS[paper_type]
    material = s["material"]
    if material == "literature":
        base = _literature_material(spec, refs, references_block, ab_text)
        if s.get("with_spec_block"):
            base += "\n\n" + _spec_material(paper_type, spec, s["with_spec_block"], s.get("spec_label", "题目"))
    elif material == "literature+spec":
        base = _spec_material(paper_type, spec, s["spec_block"], s.get("spec_label", "题目"),
                              s.get("extra_rules"))
        if refs:
            base += (f"\n\n## ⚠️ 引用编号范围：严格限定 [1]–[{len(refs)}]\n"
                     f"正文只允许出现 [1] 到 [{len(refs)}] 的编号；超出该范围即视为编造文献，"
                     "不得凭印象续编号。\n")
            base += "\n## 参考文献清单（可引用，用于文献对话；不得编造清单外文献）\n" + references_block
    else:
        return (_spec_material(paper_type, spec, s["spec_block"], s.get("spec_label", "题目"),
                               s.get("extra_rules")), stats)

    if ab_text:
        base += "\n\n" + ab_text
        base += ("\n提示：以上摘要属于文献原文，撰写「国内外研究现状」「相关工作」"
                 "「理论基础」等章节时应据此展开具体论述，而不是只按题名做空泛归类；"
                 "但无摘要的文献仍不得补造。\n")
    if gist_text:
        base += "\n\n" + gist_text
        base += ("\n提示：写「国内外研究现状」「相关工作」「理论基础」时，有概要的文献"
                 "必须依据对应编号的概要转述，不得只看题名就推断该文献得出了什么结论；"
                 "概要没写的结论一律不写。\n")
    # 人工对上一篇诊断书的批注与选材偏好，单条即生效（不必等台账攒够两次）。
    try:
        from engine import selection_ledger as SL
        _notes = SL.review_notes_for()
        if _notes:
            base += ("\n\n【人工批注：上一篇论文诊断书里人已确认要改的事，本次务必落实】\n"
                     + "\n".join(f"- {x['detail']}" for x in _notes))
        _pp = SL.preference_prompt(spec.get("title") or spec.get("topic") or "")
        if _pp:
            base += "\n\n" + _pp
    except Exception:
        pass
    return base, stats


# ── 分章写作提示 ──────────────────────────────────────────────────────
_COMMON_PRE = "请严格基于以上事实素材撰写"

TYPE_SPECS: dict[str, dict] = {
    # ══════════════════════════════════════════════════════════════════
    "survey": {
        "label": "文献综述型",
        "material": "literature",
        "field_alias": {"method_survey": "retrieval_method"},
        "cite_block_at": "intro",
        "related_work_after": None,
        "extras": [("### 7.0 本文的独立见解", "innovations"),
                   ("### 7.1 研究局限", "limitations")],
        "batches": [
            {"max_tokens": 4096, "fields": ["abstract", "keywords", "abstract_en", "keywords_en", "intro", "retrieval_method"], "prompt":
                _COMMON_PRE + "文献综述论文的前置与第 1–2 章，输出 JSON：\n"
                '{"abstract":"250-350字中文摘要（第三人称）：综述主题、文献规模、主要脉络、核心结论与开放问题",\n'
                '  "keywords":"3-4个中文关键词，用；分隔",\n'
                '  "abstract_en":"与中文摘要严格一致的英文摘要",\n'
                '  "keywords_en":"对应的英文关键词，逗号分隔",\n'
                '  "intro":"绪论正文 4 段（直接写正文，不写章标题）：①研究背景与实际/理论意义；②该领域的研究缺口与未解问题；③本文综述范围与组织方式（按主题/方法划分）；④本文的贡献（系统梳理若干主题、厘清争论、指出开放方向）",\n'
                '  "retrieval_method":"第2章「文献检索与分析方法」正文 2-3 段：检索策略（数据库/关键词/时间窗）、纳入排除标准、文献计量概况（总量、年份跨度、主题分布），须与素材中的真实数字一致"}'},
            {"max_tokens": 4096, "fields": ["framework", "status", "trend"], "prompt":
                _COMMON_PRE + "综述主体，输出 JSON：\n"
                '{"framework":"第3章「研究主题分类框架」正文 2-3 段：提出本文的综述分类维度（如按方法路线/应用场景/理论视角），说明为何这样划分，并给出各类别定义",\n'
                '  "status":"第4章「各主题研究现状」正文（不少于 1800 字，直接写正文，不写章标题）：按第3章框架逐主题综述，每个主题下按『代表工作→立场→局限→彼此关系』组织，必须广泛出现 [编号] 引用（至少覆盖 20 篇不同文献），明确指出各工作的贡献与不足，以及本文归纳出的主线。**只能定性描述方法路线与贡献方向，严禁写出任何具体性能数字（准确率/AUC/F1/敏感度等）或实验数字——你并没有这些文献的摘要与指标数据，写了即视为编造。**",\n'
                '  "trend":"第5章「研究趋势与争论」正文（不少于 800 字）：归纳近年研究趋势、不同阵营的方法论争论、尚未达成一致的关键分歧"}'},
            {"max_tokens": 3072, "fields": ["open", "conclusion", "innovations", "limitations"], "prompt":
                _COMMON_PRE + "综述的收尾部分，输出 JSON：\n"
                '{"open":"第6章「开放问题与未来方向」正文（不少于 600 字）：列出 3-5 个该领域尚未解决的关键问题，并给出有依据的未来方向",\n'
                '  "conclusion":"第7章「结论与展望」正文 3 段（不写章标题）：①复述核心发现（各主题的主线与共识）；②本文综述工作的整体价值；③可落地的后续研究建议",\n'
                '  "innovations":"本文的独立性见解（3 条，每条针对素材中的真实争论/空白，给出本文的归纳性判断；不是提出新方法，而是提出新的组织视角或判断）",\n'
                '  "limitations":"2-3 条本文综述本身的局限（如检索覆盖面、语种偏向、时效性）"}'},
        ],
        "fallback": "survey",
    },
    # ══════════════════════════════════════════════════════════════════
    "system_design": {
        "label": "系统/软件设计型",
        "material": "literature+spec",
        "spec_block": "system_spec",
        "spec_label": "系统设计",
        "field_alias": {},
        "cite_block_at": None,
        "related_work_after": "requirements",
        "extras": [("### 7.0 本文设计亮点与独立见解", "innovations"),
                   ("### 7.1 系统局限", "limitations")],
        "batches": [
            {"max_tokens": 4096, "fields": ["abstract", "keywords", "abstract_en", "keywords_en", "intro", "requirements"], "prompt":
                _COMMON_PRE + "系统/软件设计类论文的前置与第 1–2 章，输出 JSON：\n"
                '{"abstract":"250-350字中文摘要（第三人称）：系统要解决的问题、采用的核心技术/架构、主要设计与验证安排",\n'
                '  "keywords":"3-4个中文关键词，用；分隔",\n'
                '  "abstract_en":"与中文摘要严格一致的英文摘要",\n'
                '  "keywords_en":"对应的英文关键词，逗号分隔",\n'
                '  "intro":"绪论正文 3 段（直接写正文）：①问题背景与意义；②现有方案不足；③本文工作目标与范围",\n'
                '  "requirements":"第2章「需求分析」正文（不少于 600 字）：功能性需求（分点说明各模块职责）与非功能性需求（性能/可用/安全/可维护），须与素材一致"}'},
            {"max_tokens": 4096, "fields": ["architecture", "modules", "tech"], "prompt":
                _COMMON_PRE + "第 3–5 章，输出 JSON：\n"
                '{"architecture":"第3章「总体架构设计」正文（不少于 700 字）：分层/模块划分、各层职责、数据流向，须与素材中的架构描述一致",\n'
                '  "modules":"第4章「核心模块设计与实现」正文（不少于 1200 字）：逐模块说明职责、关键数据结构、接口与核心算法要点。各模块的输入输出、标签类别必须与素材 modules 描述逐字一致，不得写成不同类别或增删类别，不得虚构未声明的接口或功能。",\n'
                '  "tech":"第5章「关键技术与难点」正文（不少于 600 字）：关键技术选型理由、实现中的主要难点与应对"}'},
            {"max_tokens": 3072, "fields": ["validation", "conclusion", "innovations", "limitations"], "prompt":
                _COMMON_PRE + "第 6 章与收尾，输出 JSON：\n"
                '{"validation":"第6章「系统验证与测试」正文（不少于 600 字），仅按以下约束撰写：①只能复述素材 validation 明确给出的『策略』与『测试用例目标值（验收标准）』，目标值原文照抄；②严禁编造素材未给出的任何具体参数或测试场景数字（采样率、并发人数、测试时长、断网恢复秒数、CPU/内存占用等）；③严禁描写『实测/实验得到』的过程或结果，系统尚未部署运行，只能写『设计阶段设定以下验收标准，将在联调/试点阶段按此验证』；④可定性说明测试层级划分与每级验证目的，但不许补造数字。",\n'
                '  "conclusion":"第7章「结论与展望」正文 3 段：①系统达成的主要目标；②本文工作价值；③可落地改进方向",\n'
                '  "innovations":"本文的设计亮点/独立见解（3 条，针对真实需求与技术选型，说明设计权衡与取舍）。**亮点必须来自前文已写、素材已给的技术点**，不得新增前文未出现的指标维度、算法或组件（例如素材只给了时域指标，就不许写频域分析）",\n'
                '  "limitations":"2-3 条本系统的局限（如未覆盖的边界场景、未做性能压测等）"}'},
        ],
        "fallback": "system_design",
    },
    # ══════════════════════════════════════════════════════════════════
    "case_study": {
        "label": "案例研究型",
        "material": "literature+spec",
        "spec_block": "case_spec",
        "spec_label": "案例",
        "field_alias": {},
        "cite_block_at": "theory",
        "related_work_after": None,
        "extras": [("### 7.0 本文的分析贡献", "innovations"),
                   ("### 7.1 研究局限", "limitations")],
        "batches": [
            {"max_tokens": 4096, "fields": ["abstract", "keywords", "abstract_en", "keywords_en", "intro", "theory"], "prompt":
                _COMMON_PRE + "案例研究论文的前置与第 1–2 章，输出 JSON：\n"
                '{"abstract":"250-350字中文摘要（第三人称）：研究问题、案例对象、分析框架、主要发现与启示",\n'
                '  "keywords":"3-4个中文关键词，用；分隔",\n'
                '  "abstract_en":"与中文摘要严格一致的英文摘要",\n'
                '  "keywords_en":"对应的英文关键词，逗号分隔",\n'
                '  "intro":"绪论正文 3 段：①该现象的现实背景与研究意义；②现有研究对该问题的处理及其不足；③本文的研究问题与案例选择理由",\n'
                '  "theory":"第2章「理论基础与文献回顾」正文（不少于 800 字）：阐述本文所依据的理论框架/分析视角，并结合入选文献 [编号] 说明该理论在同类案例中的应用与分歧。对文献只能定性转述其方法与结论方向，严禁编造具体数字。"}'},
            {"max_tokens": 4096, "fields": ["design", "context", "analysis"], "prompt":
                _COMMON_PRE + "第 3–5 章，输出 JSON：\n"
                '{"design":"第3章「研究设计与案例选择」正文（不少于 600 字）：研究方法（单案例/多案例）、资料收集方式（访谈/文档/观察等，须与素材一致）、案例选择理由与典型性、资料整理与编码步骤、效度保障",\n'
                '  "context":"第4章「案例背景与现状」正文（不少于 900 字）：只写素材给出的背景、对象、现象与已提供的数据，不得补造营收、人数、比率、时间点等未给出的数字；素材未给的信息一律不写",\n'
                '  "analysis":"第5章「案例分析」正文（不少于 1500 字）：按分析框架逐维度剖析现象、机制与因果链条，以素材中的事实为证据，明确区分『素材事实』与『本文推断』"}'},
            {"max_tokens": 3072, "fields": ["discussion", "conclusion", "innovations", "limitations"], "prompt":
                _COMMON_PRE + "第 6 章与收尾，输出 JSON：\n"
                '{"discussion":"第6章「讨论与启示」正文（不少于 800 字）：与文献对话（用 [编号] 引用），说明本案例对既有理论的印证、补充或挑战，并给出可操作的实践启示与适用边界",\n'
                '  "conclusion":"第7章「结论与展望」正文 3 段：①研究问题的回答；②方法与发现的贡献；③单案例研究的局限与后续方向",\n'
                '  "innovations":"本文的分析贡献（3 条：新的分析视角、对既有解释的修正、可迁移的实践启示）",\n'
                '  "limitations":"2-3 条局限（单案例外部效度、资料可得性、研究者主观性等）"}'},
        ],
        "fallback": "case_study",
    },
    # ══════════════════════════════════════════════════════════════════
    "technical_report": {
        "label": "工程实践报告型",
        "material": "literature+spec",
        "spec_block": "project_spec",
        "spec_label": "工程项目",
        "field_alias": {},
        "cite_block_at": None,
        "related_work_after": "requirements",
        "extras": [("### 7.0 工程要点与经验", "innovations"),
                   ("### 7.1 不足与后续改进", "limitations")],
        "batches": [
            {"max_tokens": 4096, "fields": ["abstract", "keywords", "abstract_en", "keywords_en", "intro", "requirements"], "prompt":
                _COMMON_PRE + "工程实践报告的前置与第 1–2 章，输出 JSON：\n"
                '{"abstract":"250-350字中文摘要：项目背景、建设目标、采用的方案与主要交付物",\n'
                '  "keywords":"3-4个中文关键词，用；分隔",\n'
                '  "abstract_en":"与中文摘要严格一致的英文摘要",\n'
                '  "keywords_en":"对应的英文关键词，逗号分隔",\n'
                '  "intro":"第1章「概述」正文（不少于 700 字）：项目来源与业务背景、建设目标与范围、项目要解决的具体业务问题及其实际价值（据素材说清解决了什么，不写空泛口号）、本报告的组织结构",\n'
                '  "requirements":"第2章「需求与约束」正文（不少于 800 字）：业务需求、功能需求、非功能指标（须与素材一致）、实施约束（时间/成本/环境/合规）"}'},
            {"max_tokens": 4096, "fields": ["architecture", "implementation"], "prompt":
                _COMMON_PRE + "第 3–4 章，输出 JSON：\n"
                '{"architecture":"第3章「方案设计」正文（不少于 900 字）：总体方案、模块划分、接口与数据流、关键技术选型理由",\n'
                '  "implementation":"第4章「实施过程」正文（不少于 1600 字）：按阶段说明实施步骤、每步的输入输出与里程碑、遇到的实际问题与处理方式。**严禁引入 tech_stack 之外的技术组件**（数据库/缓存/消息队列/Web 服务器/ORM 及其版本号，如 Redis、Nginx、Sqoop、MySQL 5.5 等一律不得出现），不得虚构未在素材中声明的过程细节与量化数据"}'},
            {"max_tokens": 3072, "fields": ["results", "issues", "conclusion", "innovations", "limitations"], "prompt":
                _COMMON_PRE + "第 5–7 章，输出 JSON：\n"
                '{"results":"第5章「结果与验证」正文（不少于 800 字）：只能写素材给出的交付物与验收标准，按条说明如何核验；素材未给实测数据时写『按验收标准核验』，严禁编造实测指标数字",\n'
                '  "issues":"第6章「问题与改进」正文（不少于 700 字）：实施中暴露的问题、根因分析、已采取的改进与仍待优化之处",\n'
                '  "conclusion":"第7章「总结与展望」正文（不少于 500 字）：完成情况、工程价值、后续演进方向",\n'
                '  "innovations":"本项目的工程要点与可复用经验（3 条）。只能总结前文已写、素材已给的内容，不得新增未出现的技术或数字",\n'
                '  "limitations":"2-3 条不足（覆盖范围、未做的验证、遗留风险）"}'},
        ],
        "fallback": "technical_report",
    },
    # ══════════════════════════════════════════════════════════════════
    "research_proposal": {
        "label": "研究方案设计型（开题）",
        "material": "literature+spec",
        "spec_block": "proposal_spec",
        "spec_label": "研究方案",
        "field_alias": {},
        "cite_block_at": "status",
        "related_work_after": None,
        "extras": [("### 7.0 方案的创新点", "innovations"),
                   ("### 7.1 风险与应对", "limitations")],
        "batches": [
            {"max_tokens": 4096, "fields": ["abstract", "keywords", "abstract_en", "keywords_en", "intro", "status"], "prompt":
                _COMMON_PRE + "开题报告的前置与第 1–2 章，输出 JSON：\n"
                '{"abstract":"250-350字中文摘要：拟研究的问题、研究现状的空白、拟采用的方法、预期贡献",\n'
                '  "keywords":"3-4个中文关键词，用；分隔",\n'
                '  "abstract_en":"与中文摘要严格一致的英文摘要",\n'
                '  "keywords_en":"对应的英文关键词，逗号分隔",\n'
                '  "intro":"第1章「研究背景与意义」正文 3 段：①现实与学术背景；②问题的紧迫性；③理论意义与应用价值",\n'
                '  "status":"第2章「国内外研究现状」正文（不少于 900 字）：按主题梳理入选文献 [编号] 的路线与结论方向，指出尚未解决的问题（研究空白）。只能定性转述，严禁编造任何具体性能数字或实验数据。"}'},
            {"max_tokens": 4096, "fields": ["questions", "content", "route"], "prompt":
                _COMMON_PRE + "第 3–5 章，输出 JSON：\n"
                '{"questions":"第3章「研究问题与目标」正文（不少于 600 字）：提出 3-4 个可检验的研究问题（每个问题说明为何值得研究），并给出总体目标与具体目标，明确研究边界",\n'
                '  "content":"第4章「研究内容与方法」正文（不少于 1000 字）：分点列出研究内容模块，并为每块指明对应方法（须与素材一致），说明数据来源与处理方式。**严禁编造素材未给出的具体参数**：样本量、班级/人数、统计工具（SPSS/R/Python 库）、专家组构成与人数、试点规模、量表条目数等一律不写；确需说明时只能写『拟在试点中确定』",\n'
                '  "route":"第5章「技术路线与可行性」正文（不少于 700 字）：给出从问题到结论的技术路线（步骤链条）、关键难点与应对，并从理论、数据、条件三方面论证可行性。同样严禁编造素材未给出的工具、样本量与实施细节"}'},
            {"max_tokens": 3072, "fields": ["plan", "conclusion", "innovations", "limitations"], "prompt":
                _COMMON_PRE + "第 6–7 章与收尾，输出 JSON：\n"
                '{"plan":"第6章「进度安排与预期成果」正文（不少于 500 字）：按素材给出的阶段安排说明各阶段任务与产出节点，预期成果（论文/原型/数据集等）不得编造量化指标",\n'
                '  "conclusion":"第7章「结语」正文 2 段：方案要点回顾与执行承诺式说明",\n'
                '  "innovations":"本方案的创新点（3 条：问题视角、方法组合、应用场景）",\n'
                '  "limitations":"2-3 条风险与应对（数据可得性、方法适用性、进度风险）"}'},
        ],
        "fallback": "research_proposal",
    },
}


# ── 确定性兜底（网关不可用时仍产出结构完整的骨架） ────────────────────
def _fallback_survey(spec: dict, refs: list, references_block: str) -> dict:
    topic = _topic_of(spec)
    n = len(refs)
    listing = "\n".join(
        f"- [{r['num']}] {r.get('title', '')}（{r.get('year', '')}）：{r.get('venue', '')}"
        for r in refs[:30]) or "（本次未检索到文献）"
    return {
        "abstract": f"本文对「{topic}」相关研究进行系统综述，梳理 {n} 篇文献的研究脉络与主要结论。",
        "keywords": "文献综述；研究现状；方法路线",
        "abstract_en": f"This paper surveys the literature on {topic}, summarizing {n} references.",
        "keywords_en": "survey, state of the art, methods",
        "intro": (f"本文围绕「{topic}」展开文献综述。近年来该方向受到较多关注，相关工作在方法路线"
                  f"与应用场景上呈现出多样化的发展。本文旨在系统梳理已入选的 {n} 篇文献，"
                  f"厘清主要研究脉络与尚存的分歧，为后续研究提供可参照的坐标。"),
        "retrieval_method": f"本文以「{topic}」为检索主题，经筛选后纳入 {n} 篇文献，覆盖近二十年的主要工作。",
        "framework": "本文按方法路线与应用场景对各相关工作进行分类，作为后续综述的主线。",
        "status": "本文依据入选文献，按主题梳理研究现状。\n\n" + listing,
        "trend": "近年该方向在性能与可解释性之间持续权衡，不同路线的方法论争论仍未完全达成一致。",
        "open": "仍有若干开放问题待解，包括评测口径的统一、跨域泛化能力的验证等。",
        "conclusion": f"综上，本文综述表明「{topic}」已形成较为丰富的研究积累，但关键分歧仍待后续工作弥合。",
        "innovations": f"本文从「{n} 篇文献」中归纳出统一的研究主题框架，为快速把握该领域提供坐标。",
        "limitations": "本文综述覆盖面受检索策略限制，部分小语种或最新预印本可能未充分纳入。",
    }


def _fallback_system_design(spec: dict, refs: list, references_block: str) -> dict:
    topic = _topic_of(spec)
    ss = spec.get("system_spec") or spec.get("system_design") or {}
    tech = "、".join(ss.get("tech_stack", []) or []) or "通用技术栈"
    layers = "；".join((ss.get("architecture") or {}).get("layers", []) or [])
    mods = "；".join(f"{m.get('name', '')}（{m.get('responsibility', '')}）"
                    for m in (ss.get("modules") or []) if isinstance(m, dict))
    return {
        "abstract": f"本文设计并实现一个{topic}系统，采用{tech}，完成了需求分析、架构设计与核心模块实现。",
        "keywords": "系统设计；架构；模块实现",
        "abstract_en": f"This paper designs and implements {topic}.",
        "keywords_en": "system design, architecture, implementation",
        "intro": f"针对{topic}的实际需求，本文设计并实现了一套完整方案，以应对现有方案的不足。",
        "requirements": (("功能需求包括：%s。" % "；".join((ss.get("requirements") or {}).get("functional", []) or []))
                         if isinstance(ss.get("requirements"), dict) else "") or
                        "系统需满足既定的功能性需求与非功能性需求，具体以题目与领域规范为准。",
        "architecture": (f"系统采用分层架构：{layers}。" if layers else
                         "系统采用分层架构，自底向上划分为数据层、服务层与接口层，各层职责清晰、依赖单向。"),
        "modules": (f"核心模块划分如下：{mods}。" if mods else
                    "核心模块按职责划分，分别承担数据采集、业务处理与结果呈现，模块间通过明确定义的接口协作。"),
        "tech": (f"实现以{tech}为主，重点解决组件集成与数据一致性难点。"),
        "validation": "系统验证按单元测试、接口测试与系统测试三级展开，验收标准以设计阶段设定的目标值为准。",
        "conclusion": f"综上，本文完成了{topic}的设计与实现，达到了预期目标。",
        "innovations": f"本文在{topic}的架构划分与模块职责界定上做出了清晰的工程权衡。",
        "limitations": "本文未对系统进行大规模性能压测，部分边界场景的处理仍有待补充。",
    }


def _fallback_case_study(spec: dict, refs: list, references_block: str) -> dict:
    topic = _topic_of(spec)
    cs = spec.get("case_spec") or {}
    subject = cs.get("subject") or cs.get("case") or "所选案例"
    framework = cs.get("framework") or "既有理论框架"
    return {
        "abstract": f"本文以「{subject}」为案例，运用{framework}对其展开分析，提炼可迁移的实践启示。",
        "keywords": "案例研究；机制分析；实践启示",
        "abstract_en": f"This paper conducts a case study on {subject}.",
        "keywords_en": "case study, mechanism, implications",
        "intro": f"「{topic}」在现实中呈现出值得解释的现象，既有研究对此仍有讨论空间，本文据此选择{subject}作为分析对象。",
        "theory": f"本文以{framework}作为分析视角，该框架在同类案例研究中被广泛采用，但对具体机制的解释仍存在分歧。",
        "design": f"本文采用单案例研究设计，围绕{subject}收集公开资料与文档材料，按框架维度进行归类与对照分析。",
        "context": (cs.get("background") or f"本案例的基本情况依据 task_spec.json 中 case_spec 的描述，尚未补充更细的现场资料。"),
        "analysis": "按分析框架逐维度考察案例事实，梳理现象背后的机制与条件，并区分素材事实与本文推断。",
        "discussion": f"本案例对{framework}构成了局部印证，同时在边界条件上提出了需要修正之处，对同类实践具有参考价值。",
        "conclusion": "本文回答了既定的研究问题，说明了单案例研究的贡献与外部效度限制。",
        "innovations": "本文提供了该现象的一个结构化解释路径，并给出了可迁移的判断依据。",
        "limitations": "单案例研究的外部效度有限，资料可得性亦限制了机制刻画的细度。",
    }


def _fallback_technical_report(spec: dict, refs: list, references_block: str) -> dict:
    topic = _topic_of(spec)
    ps = spec.get("project_spec") or {}
    tech = "、".join(ps.get("tech_stack", []) or []) or "既定技术方案"
    return {
        "abstract": f"本报告记录「{topic}」的建设过程，包括需求分析、方案设计、实施步骤与验收情况。",
        "keywords": "工程实践；方案设计；实施；验收",
        "abstract_en": f"This report documents the engineering practice of {topic}.",
        "keywords_en": "engineering practice, solution design, implementation",
        "intro": f"本报告围绕「{topic}」的工程实践展开，说明项目来源、建设目标与实施范围。",
        "requirements": "项目需求分为业务需求与功能需求两类，并受时间、成本与环境条件约束，具体以任务书为准。",
        "architecture": f"方案以{tech}为主线，划分若干功能模块并通过明确定义的接口协作。",
        "implementation": "实施按阶段推进：环境准备、模块开发、集成联调、验收交付，各阶段均设里程碑与交付物。",
        "results": "交付物按验收标准逐条核验，核验结论以任务书约定的验收口径为准。",
        "issues": "实施过程中暴露的接口不一致与集成问题已通过统一约定与回归验证处理，仍有部分场景待优化。",
        "conclusion": f"本项目完成了{topic}的阶段性建设目标，形成了可复用的工程经验。",
        "innovations": "方案在模块边界划分与集成方式上做出了适配实际条件的取舍。",
        "limitations": "覆盖范围有限，未开展大规模压力测试，遗留风险需在后续迭代中跟踪。",
    }


def _fallback_research_proposal(spec: dict, refs: list, references_block: str) -> dict:
    topic = _topic_of(spec)
    n = len(refs)
    pr = spec.get("proposal_spec") or {}
    methods = "；".join(pr.get("methods", []) or []) or "文献研究、对比分析与实验验证"
    return {
        "abstract": f"本方案拟围绕「{topic}」展开研究，梳理 {n} 篇相关文献，明确研究问题、方法与预期成果。",
        "keywords": "研究方案；研究问题；技术路线",
        "abstract_en": f"This proposal outlines a research plan on {topic}.",
        "keywords_en": "research proposal, research questions, technical route",
        "intro": f"「{topic}」具有明确的现实需求与学术价值，本方案拟系统推进该问题的研究。",
        "status": f"已有 {n} 篇相关工作从不同路线切入该问题，但对关键机制的结论尚不一致，构成本文的研究空白。",
        "questions": "本方案提出若干可检验的研究问题，并据此设定总体目标与分项目标。",
        "content": f"研究内容按模块划分，分别对应{methods}等方法。",
        "route": "技术路线自问题界定出发，经数据获取、方法实现、结果分析到结论提炼，各环节均设校验点。",
        "plan": "研究按阶段推进，各阶段设定明确任务与产出节点，预期形成论文与可复现的实验材料。",
        "conclusion": "本方案问题清晰、路线可行、成果可期，具备按期完成的条件。",
        "innovations": "本方案在问题界定与方法组合上具有针对性，可与既有工作形成互补。",
        "limitations": "风险主要来自数据可得性与方法适用性，已设定备选方案予以应对。",
    }


_FALLBACKS = {
    "survey": _fallback_survey,
    "system_design": _fallback_system_design,
    "case_study": _fallback_case_study,
    "technical_report": _fallback_technical_report,
    "research_proposal": _fallback_research_proposal,
}


def fallback_for(paper_type: str, spec: dict, refs: list,
                 references_block: str) -> dict:
    fn = _FALLBACKS.get(TYPE_SPECS[paper_type]["fallback"])
    if fn is None:
        return {}
    return fn(spec, refs, references_block)


# ── 对外查询接口 ──────────────────────────────────────────────────────
def is_pluggable(paper_type: str) -> bool:
    return paper_type in TYPE_SPECS


def batches_for(paper_type: str) -> list[dict]:
    return TYPE_SPECS[paper_type].get("batches", [])


def field_alias_for(paper_type: str) -> dict:
    return TYPE_SPECS[paper_type].get("field_alias", {})


def extras_for(paper_type: str) -> list[tuple[str, str]]:
    return TYPE_SPECS[paper_type].get("extras", [])


def cite_block_at(paper_type: str):
    return TYPE_SPECS[paper_type].get("cite_block_at")


def related_work_after(paper_type: str):
    return TYPE_SPECS[paper_type].get("related_work_after")


def label_of(paper_type: str) -> str:
    return TYPE_SPECS.get(paper_type, {}).get("label", paper_type)


def _selftest() -> None:  # pragma: no cover - 人工自检用
    import sys
    from pathlib import Path
    _root = str(Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from paper.thesis_template import PAPER_TYPES, chapters_for
    for pt in PAPER_TYPES:
        if PAPER_TYPES[pt]["requires_experiment"]:
            continue
        chs = chapters_for(pt)
        keys = {c["key"] for c in chs}
        spec = TYPE_SPECS[pt]
        alias = spec.get("field_alias", {})
        fields = [alias.get(k, k) for k in keys]
        fb = fallback_for(pt, {}, [], "")
        missing = [f for f in fields if not str(fb.get(f, "")).strip()]
        extra = [k for k in fb if k not in fields and k not in
                 ("abstract", "keywords", "abstract_en", "keywords_en",
                  "innovations", "limitations")]
        assert not missing, f"{pt} 兜底缺字段: {missing}"
        assert not extra, f"{pt} 兜底多余字段: {extra}"
        assert len(spec.get("batches", [])) >= 1, f"{pt} 缺写作批次"
        print(f"  OK {pt:18s} chapters={len(chs)} fields={len(fields)} "
              f"batches={len(spec['batches'])} extras={len(spec.get('extras', []))}")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
