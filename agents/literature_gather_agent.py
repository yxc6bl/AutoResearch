"""文献检索 Agent（LM 驱动）—— 让 LM 真正去「搜」和「筛」。

此前 LiteratureAgent 只是「调一次 arXiv/OpenAlex API + 规则过滤」，用户指出这能力
太弱：类型受限、硬预印本为主、不会「理解」选题。本模块把检索升级为一个有 LM 参与的
Agent 回路：

  ① LM 理解选题 → 合成多条针对性检索式（基础方法 / 综述 / 本数据集应用 /
     对比基准 / 竞争路线），而不是拿一个关键词硬搜；
  ② 广撒网：对每条检索式都去 arXiv + OpenAlex 抓候选，合并去重，候选池显著扩大
     （默认上限 250，远高于旧版的 ~45）；
  ③ LM 对候选做智能筛选 + 角色归类 + 缺口识别（哪些重要文献/主题没抓到）；
  ④ 缺口回填：把 LM 指出的 missing_queries 再抓一轮，补强覆盖。

确定性兜底：网关不可用时，检索式退化为关键词派生，筛选退化为 literature_agent 的
规则打分（quality_score / venue_tier / role）。无论哪种路径，最终都产出与论文生成器
兼容的 01_literature.json（精选集）+ 02_literature_candidates.json（全量候选）
+ 0x_search_report.json（检索报告，记录「范围扩大了没有」）。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import llm_client as LLM  # noqa: E402
import literature as lit_mod  # noqa: E402
from agents import literature_agent as LA  # noqa: E402


# ── 检索式合成 ──────────────────────────────────────────────────────────
def _keyword_queries(topic: str, task_type: str, method_cn: list[str],
                     dataset_query: str) -> list[str]:
    """确定性兜底：派生**以领域词为主**的检索式（无 LM 时用）。

    旧实现把方法名单独当检索式（如 "逻辑回归 machine learning"），抓回来的
    全是通用机器学习论文，直接污染候选池。现在所有检索式都锚定领域短语，
    方法名只作为领域短语的修饰词出现。
    """
    domain = (dataset_query or "").strip()
    if not domain:
        toks = [t for t in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", topic)
                if t.lower() not in LA._FIT_GENERIC]
        domain = " ".join(toks[:5]).strip()
    qs: list[str] = []
    if domain:
        qs.append(domain)
        qs.append(f"{domain} survey review")
        qs.append(f"{domain} benchmark evaluation")
        for m in method_cn[:2]:
            qs.append(f"{domain} {m}")
        if task_type == "classification":
            qs.append(f"{domain} classification comparison")
        elif task_type == "regression":
            qs.append(f"{domain} regression prediction")
    else:
        # 连领域词都抽不出来：退回按任务类型找综述，至少保证不是空池
        qs.append(f"{task_type} machine learning benchmark survey")
    seen, uniq = set(), []
    for q in qs:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            uniq.append(q)
    return uniq[:6]


def _lm_queries(topic: str, task_type: str, method_cn: list[str],
                dataset_query: str) -> list[str]:
    """LM 据选题合成多条检索式。失败退化为关键词派生。"""
    methods = "、".join(method_cn) or "多种机器学习方法"
    prompt = (
        f"研究选题：{topic}\n"
        f"任务类型：{task_type}\n"
        f"涉及方法：{methods}\n"
        f"数据集相关检索词：{dataset_query or '（无）'}\n\n"
        "你是一位文献检索专家。请为这个选题设计 5-6 条英文文献检索式，覆盖：\n"
        "1) 核心方法的奠基/综述论文；2) 该方法在同类任务上的最新进展；\n"
        "3) 本数据集/应用领域的相关研究；4) 对比基准与竞争路线（如深度学习、其他集成方法）；\n"
        "5) 方法评估/可解释性等支撑主题。\n"
        "每条检索式要具体、可落入 arXiv 或 Google Scholar 搜索框。\n"
        '只输出 JSON：{"queries": ["检索式1", ...]}'
    )
    out = LLM.chat_json(prompt, system=LLM.FACT_FIRST_SYSTEM, temperature=0.4,
                        max_tokens=800)
    qs = out.get("queries") or []
    if not isinstance(qs, list) or not qs:
        return _keyword_queries(topic, task_type, method_cn, dataset_query)
    # 清洗 + 去重
    seen, uniq = set(), []
    for q in qs:
        q = str(q).strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            uniq.append(q)
    return uniq[:6]


# ── 候选抓取（广撒网，多源注册表） ──────────────────────────────────────
# 为什么多源：只有 arXiv（预印本，偏 CS）+ OpenAlex 时，一是学科覆盖窄（商科/医学/人文
# 的正式出版物在期刊库里），二是单源被限流就断粮。这里做成注册表，按需增删。
# 各源定位（别混）：
#   crossref —— DOI 元数据注册，期刊最全；部分会议录有 DOI，但 CS 顶会/预印本远不全覆盖。
#   arxiv    —— 预印本，CS/AI 新工作最快。
#   s2       —— Semantic Scholar，含 venue/被引，无 Key 时 429 较频繁。
#   openalex —— 覆盖面最广的开放书目，未配 mailto 时容易 429。
#   europepmc—— 生物医学专用。
#   local    —— corpus/imports/ 导入题录，不联网，唯一不会被限流的源。
# 按稳定性排序：crossref 是当前最稳的一路，放首位；易限流的放后面，
# 任一源 429/503 由 _fetch_pool 自动降权，其余源继续。
_SOURCE_FUNCS = {
    "crossref": lambda q, n: LA.fetch_crossref(q, max_results=n),
    "arxiv": lambda q, n: lit_mod.fetch_arxiv(q, max_results=n),
    "s2": lambda q, n: LA.fetch_semantic_scholar(q, max_results=n),
    "openalex": lambda q, n: LA.fetch_openalex(q, max_results=n),
    "europepmc": lambda q, n: LA.fetch_europepmc(q, max_results=n),
    "local": lambda q, n: LA.fetch_imported(q, max_results=n),
}
_SOURCE_DEFAULT = "crossref,arxiv,s2,openalex,europepmc,local"


def enabled_sources() -> list[str]:
    """可用数据源：env LIT_SOURCES 覆盖；默认六源全开，单源挂了自动降权。"""
    raw = os.environ.get("LIT_SOURCES", _SOURCE_DEFAULT)
    names = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return [n for n in names if n in _SOURCE_FUNCS] or ["arxiv"]


def _fetch_pool(queries: list[str], per_query: int = 40, deadline: float | None = None,
                saturate_rounds: int = 2, saturate_min_new: int = 5) -> list[dict]:
    """对每条检索式跑所有启用数据源，合并去重，返回带评分的候选池。

    - deadline：墙上时间预算，到点即停。
    - 饱和度停止：连续 N 条检索式新增都少于阈值，说明该题文献池已接近饱和。
    - 单源降级：某源 429/持续失败只跳过该源，其余源继续；**失败不被误判为饱和**。
    """
    raw: list[dict] = []
    seen_ids: set[str] = set()
    dry = 0
    fails = 0
    down: dict[str, str] = {}
    sources = enabled_sources()
    print(f"[GatherAgent] 数据源：{'、'.join(sources)}")

    for i, q in enumerate(queries):
        if deadline and time.time() > deadline:
            print(f"[GatherAgent] 已达检索时间预算，跳过剩余 {len(queries) - i} 条检索式"
                  f"（已抓 {len(raw)} 条候选，继续后续筛选）")
            break
        _before = len(seen_ids)
        _ok = False
        for name in sources:
            if name in down:
                continue
            fn = _SOURCE_FUNCS[name]
            n = max(8, per_query // 3) if name == "arxiv" else per_query
            try:
                _got = fn(q, n)
                raw += _got
                _ok = _ok or bool(_got)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "429" in msg or "Too Many" in msg:
                    down[name] = "429"
                    print(f"[GatherAgent] {name} 触发限流(429)，本次运行内跳过该源")
                elif "403" in msg or "401" in msg:
                    down[name] = "auth"
                    print(f"[GatherAgent] {name} 拒绝访问，本次运行内跳过该源")
                else:
                    print(f"[GatherAgent] {name} 检索失败（{q[:36]}）：{e}")
        if not _ok:
            fails += 1
            if fails >= 3:
                print("[GatherAgent] 连续 3 条检索式全部数据源失败（网络/限流），停止抓取")
                break
            continue
        fails = 0
        for p in raw:
            pid = p.get("id") or p.get("url")
            if pid:
                seen_ids.add(pid)
        _new = len(seen_ids) - _before
        dry = dry + 1 if _new < saturate_min_new else 0
        if dry >= saturate_rounds:
            print(f"[GatherAgent] 文献池已饱和（连续 {dry} 条检索式新增 < {saturate_min_new} 条），"
                  f"提前停止；累计 {len(seen_ids)} 条候选")
            break

    # 相关性过滤 + 评分 + 去重
    before = len(raw)
    raw = [p for p in raw if LA._relevance_passes(p, "")]
    seen = {}
    for p in raw:
        pid = p.get("id") or p.get("url")
        if not pid or pid in seen:
            continue
        _u = str(p.get("url", ""))
        p["source"] = ("arxiv" if "arxiv.org" in _u else
                       "crossref" if "doi.org" in _u else
                       "europepmc" if "europepmc" in _u else "openalex")
        p["venue_tier"] = LA._venue_tier(p.get("venue"), p.get("url"))
        p["role"] = LA._classify_role(p)
        p["quality_score"] = LA._quality_score(p)
        seen[pid] = p
    pool = sorted(seen.values(), key=lambda x: x.get("quality_score", 0), reverse=True)
    print(f"[GatherAgent] 候选池：{before} → 过滤去重后 {len(pool)} 篇")
    return pool


# ── LM 逐篇阅读筛选（核心：AI 读文献，而不是匹配关键词） ──────────────────
# 每批 12 篇，把「标题 + 摘要（最多 600 字）+ 出处」交给模型，逐篇回答：
#   what      这篇做了什么（≤40 字）
#   relation  与选题的关系：直接可比 / 方法可借鉴 / 背景 / 无关 / 未知
#   usable_in 能用在论文哪一节：相关工作 / 方法 / 实验对比 / 引言 / 无
#   why       判断依据（≤25 字）
# 结果按 id 缓存到 0x_screen_cache.json，重跑不重复花钱。
_REL_SCORE = {"直接可比": 0.95, "方法可借鉴": 0.72, "背景": 0.40, "未知": 0.30, "无关": 0.0}
_ROLE_FROM_USE = {"相关工作": "方法", "方法": "方法", "实验对比": "基准数据集", "引言": "综述", "无": "方法"}


def _lm_read_screen(topic: str, pool: list[dict], out_dir: Path,
                    max_papers: int = 72, batch: int = 12,
                    max_calls: int = 8, abstract_chars: int = 600,
                    deadline: float | None = None) -> dict:
    """让 LM 逐篇读摘要并给出可用性判断。返回 {id: verdict} 与统计。

    失败/预算耗尽时返回已读到的部分——调用方按「已读优先、未读退回关键词」处理，
    并据此标注 confidence。
    """
    # 成本可调：LIT_READ_MAX=逐篇阅读篇数上限（0=关闭），LIT_READ_CALLS=调用次数上限
    try:
        max_papers = int(os.environ.get("LIT_READ_MAX", max_papers))
        max_calls = int(os.environ.get("LIT_READ_CALLS", max_calls))
    except Exception:
        pass
    if max_papers <= 0:
        return {}
    cache_path = out_dir / "0x_screen_cache.json"
    cache: dict = {}
    try:
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        cache = {}

    # 优先读「关键词分较高」的候选，控制成本
    ordered = sorted(pool, key=lambda x: x.get("select_score") or x.get("quality_score") or 0,
                     reverse=True)[:max_papers]
    todo = [p for p in ordered if p.get("id") and p["id"] not in cache]
    calls = 0
    failed = 0
    while todo and calls < max_calls:
        if deadline and time.time() > deadline:
            print("[GatherAgent] 已到时间预算，停止逐篇阅读（未读的退回关键词判断）")
            break
        chunk, todo = todo[:batch], todo[batch:]
        items = [{
            "id": p["id"],
            "title": (p.get("title") or "")[:150],
            "venue": (p.get("venue") or "")[:50],
            "year": p.get("year"),
            "abstract": (p.get("abstract") or "")[:abstract_chars] or "（无摘要）",
        } for p in chunk]
        prompt = (
            f"研究选题：{topic}\n\n"
            "你是这篇本科毕业设计的文献筛选人。请逐篇**读摘要**，判断它能否用于这篇论文，"
            "而不是只做关键词匹配。对每篇输出：\n"
            '  "what"：它做了什么（用了什么方法/数据、得到什么结论），≤40 个汉字；\n'
            '  "relation"：与选题的关系，只能是 直接可比 / 方法可借鉴 / 背景 / 无关 / 未知；\n'
            '    直接可比=同一个任务/同一类数据上的同类研究；方法可借鉴=方法或思路能迁移过来；\n'
            '    背景=只提供领域背景；无关=主题不同；未知=摘要不足以判断。\n'
            '  "usable_in"：能用在论文哪一节，只能是 相关工作 / 方法 / 实验对比 / 引言 / 无；\n'
            '  "why"：判断依据，≤25 个汉字。\n'
            "硬约束：只依据给出的标题与摘要，不得臆测；摘要为空且标题不足以判断时 relation=未知。\n"
            '只输出 JSON：{"verdicts":[{"id":"...","what":"...","relation":"直接可比",'
            '"usable_in":"相关工作","why":"..."}]}'
        )
        try:
            out = LLM.chat_json(prompt + "\n\n候选文献：\n" + json.dumps(items, ensure_ascii=False),
                                system=LLM.FACT_FIRST_SYSTEM, temperature=0.2, max_tokens=2400)
            calls += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[GatherAgent] 阅读批次失败（第 {failed} 次）：{e}")
            if failed >= 2:
                break
            continue
        got = 0
        for v in (out.get("verdicts") or []):
            if not isinstance(v, dict) or not v.get("id"):
                continue
            cache[v["id"]] = {
                "what": str(v.get("what") or "")[:80],
                "relation": v.get("relation") if v.get("relation") in _REL_SCORE else "未知",
                "usable_in": v.get("usable_in") or "无",
                "why": str(v.get("why") or "")[:60],
            }
            got += 1
        if got == 0:
            failed += 1
            if failed >= 2:
                break
        else:
            failed = 0
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[GatherAgent] 逐篇阅读：已判 {len(cache)} 篇，本轮调用 {calls} 次")
    return cache


def _apply_screen(pool: list[dict], screen: dict) -> int:
    """把阅读结论写回候选：relation/usable_in/what → fit_score / role / gist。

    两个字段必须留在产物里，不能只留在 0x_screen_cache.json：
      llm_usable_in —— 读者判的「能用在论文哪一节」，审评与第 2 章分组都要用；
      confidence    —— 这篇的判断到底来自逐篇阅读、LM 语义筛选，还是关键词兜底。
    以前 usable_in 只被用来推导 role 就丢掉了，等于「AI 读了却说不清用在哪」。
    """
    n = 0
    for p in pool:
        v = screen.get(p.get("id"))
        if not v:
            # 未被逐篇阅读的：先标关键词级；后面若被 LM 语义筛选认可，再升级为 lm_semantic
            p.setdefault("confidence", "keyword_only")
            continue
        p["llm_relation"] = v["relation"]
        p["llm_what"] = v["what"]
        p["llm_why"] = v["why"]
        p["llm_usable_in"] = v.get("usable_in") or "无"
        p["confidence"] = "lm_read"
        p["llm_fit"] = _REL_SCORE.get(v["relation"], 0.3)
        p["role"] = _ROLE_FROM_USE.get(v.get("usable_in"), p.get("role") or "方法")
        if v.get("what"):
            p["gist_cn"] = v["what"]        # 直接作为正文里的一句话概要
        n += 1
    return n


# ── LM 智能筛选 + 缺口识别 ──────────────────────────────────────────────
def _lm_filter(topic: str, pool: list[dict]) -> dict:
    """LM 对候选做 keep/drop + 角色 + 缺口。返回 {keep_ids, role_map, gaps, missing_queries}。"""
    # 控制喂给 LM 的体量：取质量分 top 150 进行评估，避免超 token
    view = pool[:150]
    items = []
    for p in view:
        items.append({
            "id": p.get("id"),
            "title": (p.get("title") or "")[:140],
            "year": p.get("year"),
            "venue": (p.get("venue") or "")[:40],
            "q": p.get("quality_score"),
            "abs": (p.get("abstract") or "")[:160],
        })
    prompt = (
        f"研究选题：{topic}\n\n"
        f"下面是候选文献（共 {len(view)} 篇，已按质量分排序）。请为这篇本科毕设挑出最相关、"
        "最权威、覆盖不同技术路线的文献（建议入选 30-50 篇），并做三件事：\n"
        "1) 对每篇给 keep（true/false）；\n"
        "2) 给保留的文献一个 role：综述 / 基准数据集 / 理论基础 / 方法 / 应用；\n"
        "3) 指出当前集合的 coverage gaps（还缺哪类重要文献/主题）与 missing_queries"
        "（为补强覆盖应再搜哪些检索式）。\n"
        '只输出 JSON：{"decisions":[{"id":"...","keep":true,"role":"方法"}],'
        '"gaps":["..."],"missing_queries":["..."]}'
    )
    try:
        out = LLM.chat_json(
            prompt + "\n\n候选文献：\n" + json.dumps(items, ensure_ascii=False),
            system=LLM.FACT_FIRST_SYSTEM, temperature=0.3, max_tokens=2600)
    except Exception as e:  # noqa: BLE001
        print(f"[GatherAgent] LM 筛选失败，转规则兜底：{e}")
        return {"keep_ids": None, "role_map": {}, "gaps": [], "missing_queries": []}
    decisions = out.get("decisions") or []
    keep_ids = set()
    role_map = {}
    for d in decisions:
        if not isinstance(d, dict):
            continue
        i = d.get("id")
        if i is None:
            continue
        if d.get("keep"):
            keep_ids.add(i)
            r = d.get("role")
            if r:
                role_map[i] = r
    return {
        "keep_ids": keep_ids if keep_ids else None,
        "role_map": role_map,
        "gaps": out.get("gaps") or [],
        "missing_queries": out.get("missing_queries") or [],
    }


def _rules_select(pool: list[dict], top_n: int) -> list[dict]:
    """确定性兜底：复用 literature_agent 的规则精选。"""
    return LA.select(pool, mode="auto", top_n=top_n)


def _add_gists(topic: str, papers: list[dict], max_papers: int = 40) -> int:
    """为入选文献生成一句中文概要（≤40 字）：它做了什么。

    只做客观转述，不评价优劣、不补充原文没有的信息；摘要为空且标题不足以判断时
    返回空串，正文退回「标题 + 出处」——宁可少信息，不可编。
    """
    todo = [p for p in papers if not p.get("gist_cn")][:max_papers]
    if not todo or not LLM.available():
        return 0
    items = [{"i": i, "t": (p.get("title") or "")[:130],
              "a": (p.get("abstract") or "")[:200],
              "v": (p.get("venue") or "")[:40], "y": p.get("year")}
             for i, p in enumerate(todo)]
    prompt = (
        f"研究选题：{topic}\n\n"
        "下面是从检索结果中选中的文献。请为每篇写一句中文说明：它做了什么（方法/数据/结论要点），"
        "不超过 40 个汉字，客观转述，禁止评价优劣、禁止补充原文没有的信息；"
        "摘要为空且仅凭标题无法判断时返回空字符串。\n"
        '只输出 JSON：{"gists": {"0": "……", "1": "……"}}'
    )
    try:
        out = LLM.chat_json(prompt + "\n\n文献：\n" + json.dumps(items, ensure_ascii=False),
                            system=LLM.FACT_FIRST_SYSTEM, temperature=0.2, max_tokens=2000)
    except Exception as e:  # noqa: BLE001
        print(f"[GatherAgent] 文献概要生成失败，正文退回「标题+出处」：{e}")
        return 0
    gists = out.get("gists") or {}
    n = 0
    for k, v in gists.items():
        try:
            idx = int(k)
        except Exception:
            continue
        if 0 <= idx < len(todo) and isinstance(v, str) and v.strip():
            todo[idx]["gist_cn"] = v.strip()[:80]
            n += 1
    print(f"[GatherAgent] 文献概要：{n}/{len(todo)} 条")
    return n


def run(topic: str, task_type: str = "classification", method_cn: list[str] | None = None,
        dataset_query: str = "", out_dir: str | Path = "", top_n: int = 45,
        max_candidates: int = 250, source_tag: str = "",
        budget_sec: float = 300) -> dict:
    """LM 驱动文献检索主入口。

    budget_sec：检索阶段的墙上时间预算（默认 300s）。接口/网关慢时到点即止，
    用已抓候选继续筛选，避免「一个慢接口拖死整条流水线」。

    返回 dict：{papers(精选), candidates(全量), report(检索报告), lm_driven(bool)}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    method_cn = method_cn or []
    lm_driven = LLM.available()
    _deadline = time.time() + max(30.0, float(budget_sec))

    # ① 检索式
    if lm_driven:
        queries = _lm_queries(topic, task_type, method_cn, dataset_query)
    else:
        queries = _keyword_queries(topic, task_type, method_cn, dataset_query)
    print(f"[GatherAgent] 检索式（{'LM 合成' if lm_driven else '关键词派生'}）：{queries}")

    # 系统自我反馈闭环：上一轮论文诊断 / 文献审评指出的缺口已存成检索式，
    # 这次抓取直接并入计划，并在 LIT_PLAN 里标明这几条的来源。
    _fb_queries: list[dict] = []
    try:
        from engine import selection_ledger as SL
        _fb_queries = SL.take_feedback_queries(limit=4)
        _extra = [x["query"] for x in _fb_queries if x["query"] not in queries]
        if _extra:
            queries = queries + _extra
            print(f"[检索规划] 并入 {len(_extra)} 条历史反馈检索式：{_extra}")
    except Exception as _e:  # noqa: BLE001
        print(f"[检索规划] 反馈并入跳过：{_e}")

    # 检索规划台账：选题 → 检索式（用途/来源）→ 数据源 → 每轮结果，全程可查。
    _plan = None
    try:
        from engine import lit_plan as LP
        _plan = LP.create(topic, queries, enabled_sources(), task_type,
                          method_cn, dataset_query, lm_driven, _fb_queries)
    except Exception as _e:  # noqa: BLE001
        print(f"[检索规划] 台账初始化跳过：{_e}")

    # ② 广撒网
    pool = _fetch_pool(queries, per_query=40, deadline=_deadline)
    pool = pool[:max_candidates]

    # 历史使用反馈：过去真正被正文引用过的文献优先（同类选题复用已验证的好文献）
    try:
        from engine import lit_feedback as LFB  # 同源模块
        bm = LFB.bonus_map()
        if bm:
            hit = 0
            for p in pool:
                b = bm.get(LFB._norm_title(p.get("title")))
                if b:
                    p["score_bonus"] = float(p.get("score_bonus") or 0) + b
                    hit += 1
            if hit:
                print(f"[GatherAgent] 历史反馈：{hit} 篇曾被引用的文献获得优先加权")
    except Exception as e:  # noqa: BLE001
        print(f"[GatherAgent] 反馈加权跳过（不影响检索）：{e}")

    report = {
        "topic": topic,
        "lm_driven": lm_driven,
        "queries": queries,
        "n_candidates": len(pool),
        "gaps": [],
        "missing_queries": [],
        "rounds": 1,
    }

    # ③-a LM 逐篇阅读：让模型读摘要判断「做了什么 / 能不能用」，而不是关键词命中
    _screen: dict = {}
    if lm_driven:
        try:
            _screen = _lm_read_screen(topic, pool, out_dir, deadline=_deadline)
            _n_applied = _apply_screen(pool, _screen) if _screen else 0
            if _n_applied:
                _rel = {}
                for p in pool:
                    r = p.get("llm_relation")
                    if r:
                        _rel[r] = _rel.get(r, 0) + 1
                print(f"[GatherAgent] 阅读结论已应用 {_n_applied} 篇：{_rel}")
        except Exception as e:  # noqa: BLE001
            print(f"[GatherAgent] 逐篇阅读跳过（不影响主流程）：{e}")

    # ③ LM 判定 keep / role / 缺口（只影响排序加分，不再作为硬过滤）
    filt = (_lm_filter(topic, pool) if lm_driven
            else {"keep_ids": None, "role_map": {}, "gaps": [], "missing_queries": []})
    if lm_driven and filt.get("keep_ids") is not None:
        for p in pool:
            if p.get("id") in filt["keep_ids"]:
                p["score_bonus"] = 0.12          # LM 认可：排序加分
                # 没逐篇读过、但被 LM 语义筛选认可：可信度介于两者之间
                if p.get("confidence") == "keyword_only":
                    p["confidence"] = "lm_semantic"
                r = filt["role_map"].get(p.get("id"))
                if r:
                    p["role"] = r
        report["gaps"] = filt.get("gaps", [])
        report["missing_queries"] = filt.get("missing_queries", [])

        # ④ 缺口回填：再抓一轮 missing_queries（同样受时间预算约束）
        mq = [q for q in (filt.get("missing_queries") or []) if isinstance(q, str) and q.strip()]
        if mq:
            report["rounds"] = 2
            extra = _fetch_pool(mq[:4], per_query=30, deadline=_deadline)
            known = {p.get("id") for p in pool}
            added = [p for p in extra if p.get("id") not in known]
            if added:
                # 回填来的文献同样要逐篇读：此前这里只做 LM keep 判定，
                # 结果实测 45 篇里只有 28 篇带 llm_what，剩下全是关键词兜底。
                if lm_driven:
                    try:
                        _sc2 = _lm_read_screen(topic, added, out_dir,
                                               deadline=_deadline, max_calls=2)
                        if _sc2:
                            _screen.update(_sc2)
                            _n2 = _apply_screen(added, _sc2)
                            print(f"[GatherAgent] 缺口回填补读 {_n2} 篇")
                    except Exception as e:  # noqa: BLE001
                        print(f"[GatherAgent] 缺口回填补读跳过：{e}")
                ef = _lm_filter(topic, added)
                for p in added:
                    if ef.get("keep_ids") is None or p.get("id") in ef["keep_ids"]:
                        p["score_bonus"] = 0.12 if ef.get("keep_ids") is not None else 0.0
                        r = (ef.get("role_map") or {}).get(p.get("id"))
                        if r:
                            p["role"] = r
                pool += added
            report["n_candidates"] = len(pool)

    # ⑤ 角色覆盖式精选：数量由「每类最低覆盖 + 质量线」决定，不凑满固定篇数。
    # 贴合度必须用「中文题目 + 英文数据集检索词」共同计算：中文题目本身没有英文领域词，
    # 只用它会退化成所有候选同分（这也是之前边缘文献混进来的原因之一）。
    _fit_topic = " ".join(x for x in (topic, dataset_query) if x)
    # LM 在场时，它的 keep 是语义判断（能分清「收入预测」和「收入与健康」），
    # 作为过滤条件使用；关键词兜底分不清任务语义，因此少选、并标注置信度。
    _lm_kept = [p for p in pool if p.get("id") in filt.get("keep_ids")] \
        if (lm_driven and filt.get("keep_ids")) else []
    if _screen:
        # 语义筛选：读者判为「无关」的直接排除；未读到的保留交给关键词兜底
        _cand = [p for p in pool if p.get("llm_relation") != "无关"]
        if len(_cand) < 8:
            _cand = pool
    else:
        _cand = _lm_kept if len(_lm_kept) >= 15 else pool
    _cap = top_n if _lm_kept else min(top_n, 30)
    # 门槛由召回策略决定（LIT_PROFILE=strict/balanced/recall），不再在这里写死
    chosen, sel_report = LA.select_by_role(_cand, _fit_topic, max_total=_cap)
    if not chosen:  # 全部低于质量线时退回放宽一档，保证流水线不空转
        chosen, sel_report = LA.select_by_role(_cand, _fit_topic, max_total=_cap,
                                               profile="recall")
        sel_report["fallback"] = "全部候选低于质量线，已放宽门槛（结果需人工复核）"
    sel_report["confidence"] = ("lm_read" if _screen else
                                "lm_semantic" if _lm_kept else "keyword_only")
    sel_report["cap"] = _cap
    sel_report["n_read"] = len(_screen)
    # ⑤-b 覆盖不足时换更宽的检索式再抓一轮（扩面，不降门槛）
    _TARGET_MIN = 20
    if len(chosen) < min(_TARGET_MIN, _cap):
        _doms = LA._domain_tokens(_fit_topic)[:3]
        _wide = []
        if _doms:
            _d = " ".join(_doms)
            _wide = [f"{_d} machine learning", f"{_d} survey", f"{_d} benchmark"]
        else:
            _wide = [f"{task_type} tabular data benchmark"]
        try:
            _extra = _fetch_pool(_wide[:3], per_query=40, deadline=_deadline)
            _known = {p.get("id") or p.get("url") for p in pool}
            _added = [p for p in _extra if (p.get("id") or p.get("url")) not in _known]
            if _added:
                if lm_driven:
                    try:
                        _sc3 = _lm_read_screen(topic, _added, out_dir,
                                               deadline=_deadline, max_calls=2)
                        if _sc3:
                            _screen.update(_sc3)
                            _apply_screen(_added, _sc3)
                    except Exception as e:  # noqa: BLE001
                        print(f"[GatherAgent] 扩面候选补读跳过：{e}")
                pool += _added
                chosen, sel_report = LA.select_by_role(pool, _fit_topic, max_total=_cap)
                sel_report["confidence"] = ("lm_read" if _screen else
                                            "lm_semantic" if _lm_kept else "keyword_only")
                sel_report["cap"] = _cap
                report["broadened_queries"] = _wide[:3]
                if _plan is not None:
                    LP.add_round(_plan, "broadening", _wide[:3], len(_added),
                                 "首轮覆盖不足，按放宽主题扩面重抓")
                report["rounds"] = int(report.get("rounds") or 1) + 1
                report["n_candidates"] = len(pool)
                print(f"[GatherAgent] 覆盖不足，已扩面重抓 {len(_added)} 条新候选 → 精选 {len(chosen)} 篇")
        except Exception as e:  # noqa: BLE001
            print(f"[GatherAgent] 扩面检索跳过：{e}")

    # ⑤-c 下限兜底：严格筛选后不足 8 篇时逐级放宽，并明确标注需人工复核。
    # 1 篇文献写不出综述章，但"凑到 45 篇噪声"更糟；这里的取舍是「能写 + 标红」。
    _MIN_REFS = 8
    if len(chosen) < _MIN_REFS:
        _ch2, _rep2 = LA.select_by_role(_cand, _fit_topic,
                                        max_total=max(_MIN_REFS, len(chosen)),
                                        profile="recall")
        if len(_ch2) > len(chosen):
            chosen, sel_report = _ch2, _rep2
            sel_report["relaxed"] = "严格门槛下不足 8 篇，已放宽到 0.10"
        sel_report["confidence"] = "low"
        sel_report["needs_review"] = True

    # ⑤-d 入选补读：最终写进正文的文献，尽量每一篇都经过逐篇阅读。
    # 「读了候选 28 篇、精选 45 篇」会留下 17 篇关键词兜底混进正文——实测吃过这个亏。
    # 补读只针对最终入选且尚未读过的，目标明确、数量可控，不浪费在几百条候选上。
    if lm_driven and chosen:
        _todo = [p for p in chosen if not (p.get("llm_what") or p.get("llm_fit") is not None)]
        if _todo:
            try:
                _sc4 = _lm_read_screen(topic, _todo[:24], out_dir,
                                       deadline=_deadline + 60,
                                       max_papers=24, batch=12, max_calls=2)
                if _sc4:
                    _screen.update(_sc4)
                    _n4 = _apply_screen(_todo[:24], _sc4)
                    print(f"[GatherAgent] 入选补读 {_n4}/{len(_todo)} 篇（此前为关键词兜底）")
            except Exception as e:  # noqa: BLE001
                print(f"[GatherAgent] 入选补读跳过：{e}")

    # 跨运行人工偏好：只剔除/加分，不重排顺序——顺序就是正文 [N] 引用编号。
    _applied: dict = {}
    try:
        from engine import selection_ledger as SL
        _pp = SL.preference_prompt(topic)
        if _pp:
            print("[选材台账] 检测到人工偏好：\n" + _pp)
        chosen, _applied = SL.apply_preferences(chosen, topic)
        if _applied.get("removed"):
            print(f"[选材台账] 按人工偏好剔除 {len(_applied['removed'])} 篇")
        if _applied.get("n_after", len(chosen)) < 8:
            print("[选材台账] [警告] 应用偏好后入选不足 8 篇，正文可能缺文献；"
                  "请用 lit-pref --list 检查排除规则")
        report["preferences"] = _applied
    except Exception as _e:  # noqa: BLE001
        print(f"[选材台账] 偏好应用跳过：{_e}")

    report["selection"] = sel_report

    if sel_report.get("role_stats"):
        _short = {r: v["short"] for r, v in sel_report["role_stats"].items() if v.get("short")}
        print(f"[GatherAgent] 角色覆盖：" +
              "、".join(f"{r}{v['taken']}/{v['quota']}" for r, v in sel_report["role_stats"].items()))
        if _short:
            print(f"[GatherAgent] 覆盖缺口（如实报告，不凑数）：{_short}")
        _d = sel_report["dropped"]
        print(f"[GatherAgent] 召回策略 {sel_report.get('profile')}｜剔除："
              f"低于贴合度门槛 {_d.get('低于贴合度门槛', 0)}、重复 {_d.get('重复', 0)} 篇")
        _nm = sel_report.get("near_miss") or []
        if _nm:
            print("[GatherAgent] 差一点被剔除（可切 LIT_PROFILE=recall 收进来）：")
            for x in _nm[:3]:
                print(f"    · fit={x['fit']:.2f} 《{x['title'][:62]}》")

    report["n_selected"] = len(chosen)
    # ⑥ 为入选文献补一句话概要（失败则跳过，正文退回标题+出处）
    report["n_gists"] = _add_gists(topic, chosen)

    # 写产物
    papers_path = out_dir / "01_literature.json"
    cand_path = out_dir / "02_literature_candidates.json"
    report_path = out_dir / "0x_search_report.json"
    papers_path.write_text(json.dumps({"papers": chosen}, ensure_ascii=False, indent=2), encoding="utf-8")
    cand_path.write_text(json.dumps(
        [{"id": c.get("id"), "title": c.get("title"), "venue": c.get("venue"),
          "year": c.get("year"), "source": c.get("source"),
          "quality_score": c.get("quality_score"), "url": c.get("url"),
          "role": c.get("role")} for c in pool], ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if _plan is not None:
        try:
            LP.finish(_plan, len(pool), len(chosen), report.get("gaps") or [])
            LP.write(out_dir, _plan)
            print(f"[检索规划] 计划与结果已落盘 → {LP.NAME}")
        except Exception as _e:  # noqa: BLE001
            print(f"[检索规划] 台账写入跳过：{_e}")

    # 选材台账：候选池里每一篇的去留、理由、决策者（model/rule/human）都要留痕，
    # 否则用户事后想改「为什么把这篇选进来」没有任何依据。
    try:
        from engine import selection_ledger as SL
        _led = SL.record_selection(topic, pool, chosen, out_dir,
                                   selector=("model" if lm_driven else "rule"),
                                   applied=_applied)
        print(f"[选材台账] {_led['n_candidates']} 候选 / {_led['n_selected']} 入选"
              f" / {_led['n_human']} 篇有人工表态 → {SL.LEDGER_NAME}")
    except Exception as _e:  # noqa: BLE001
        print(f"[选材台账] 写入失败（不影响本次抓取）：{_e}")

    print(f"[GatherAgent] 精选 {len(chosen)} 篇 → {papers_path.name}（LM 驱动={lm_driven}）")
    return {"papers": chosen, "candidates": pool, "report": report, "lm_driven": lm_driven}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LM 驱动文献检索 Agent")
    ap.add_argument("--topic", required=True, help="研究选题/题目")
    ap.add_argument("--task", default="classification")
    ap.add_argument("--methods", default="", help="方法中文名，逗号分隔")
    ap.add_argument("--dataset-query", default="")
    ap.add_argument("--out", default="out_gather", help="输出目录")
    ap.add_argument("--top-n", type=int, default=45)
    args = ap.parse_args()
    mc = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else []
    r = run(args.topic, args.task, mc, args.dataset_query, args.out, top_n=args.top_n)
    print(json.dumps(r["report"], ensure_ascii=False, indent=2))
