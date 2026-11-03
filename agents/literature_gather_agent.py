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
    """确定性兜底：从选题关键词派生检索式（无 LM 时用）。"""
    qs = []
    if dataset_query:
        qs.append(dataset_query)
    for m in method_cn[:3]:
        qs.append(f"{m} machine learning")
    if task_type == "classification":
        qs.append("classification benchmark comparison survey")
    elif task_type == "regression":
        qs.append("regression prediction machine learning survey")
    if topic:
        # 题目本身的关键英文词（粗略去停用词）
        toks = re.findall(r"[a-zA-Z][a-zA-Z0-9\- ]{2,}", topic)
        if toks:
            qs.append(" ".join(toks[:6]).strip())
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


# ── 候选抓取（广撒网） ──────────────────────────────────────────────────
def _fetch_pool(queries: list[str], per_query: int = 40, deadline: float | None = None) -> list[dict]:
    """对每条检索式抓 arXiv + OpenAlex，合并去重，返回带评分的候选池。

    deadline（time.time() 绝对时刻）用于兜住「网关/接口慢」的极端情况：到点即停，
    用已抓到的候选继续走筛选，避免整条流水线被检索阶段拖死。
    """
    raw: list[dict] = []
    for i, q in enumerate(queries):
        if deadline and time.time() > deadline:
            print(f"[GatherAgent] 已达检索时间预算，跳过剩余 {len(queries) - i} 条检索式"
                  f"（已抓 {len(raw)} 条候选，继续后续筛选）")
            break
        if "arxiv.org" in q.lower():
            pass
        # arXiv
        try:
            raw += lit_mod.fetch_arxiv(q, max_results=max(8, per_query // 3))
        except Exception as e:  # noqa: BLE001
            print(f"[GatherAgent] arXiv 检索失败（{q[:40]}）：{e}")
        # OpenAlex
        try:
            raw += LA.fetch_openalex(q, max_results=per_query)
        except Exception as e:  # noqa: BLE001
            print(f"[GatherAgent] OpenAlex 检索失败（{q[:40]}）：{e}")
        # Semantic Scholar（可选，默认关闭；env AUTORESEARCH_USE_SEMSCHOLAR=1 启用，扩大正式论文覆盖）
        if os.environ.get("AUTORESEARCH_USE_SEMSCHOLAR"):
            try:
                raw += LA.fetch_semantic_scholar(q, max_results=per_query)
            except Exception as e:  # noqa: BLE001
                print(f"[GatherAgent] SemanticScholar 检索失败（{q[:40]}）：{e}")
    # 相关性过滤 + 评分 + 去重
    before = len(raw)
    raw = [p for p in raw if LA._relevance_passes(p, "")]
    seen = {}
    for p in raw:
        pid = p.get("id") or p.get("url")
        if not pid or pid in seen:
            continue
        p["source"] = "arxiv" if "arxiv.org" in str(p.get("url", "")) else "openalex"
        p["venue_tier"] = LA._venue_tier(p.get("venue"), p.get("url"))
        p["role"] = LA._classify_role(p)
        p["quality_score"] = LA._quality_score(p)
        seen[pid] = p
    pool = sorted(seen.values(), key=lambda x: x.get("quality_score", 0), reverse=True)
    print(f"[GatherAgent] 候选池：{before} → 过滤去重后 {len(pool)} 篇")
    return pool


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

    # ② 广撒网
    pool = _fetch_pool(queries, per_query=40, deadline=_deadline)
    pool = pool[:max_candidates]

    report = {
        "topic": topic,
        "lm_driven": lm_driven,
        "queries": queries,
        "n_candidates": len(pool),
        "gaps": [],
        "missing_queries": [],
        "rounds": 1,
    }

    # ③ LM 筛选 + 缺口
    filt = _lm_filter(topic, pool) if lm_driven else {"keep_ids": None, "role_map": {}, "gaps": [], "missing_queries": []}
    if lm_driven and filt.get("keep_ids") is not None:
        chosen = [p for p in pool if p.get("id") in filt["keep_ids"]]
        # 应用 LM 给的角色（覆盖规则归类）
        for p in chosen:
            r = filt["role_map"].get(p.get("id"))
            if r:
                p["role"] = r
        # 少于 top_n 时，用质量分补满，保证数量
        if len(chosen) < top_n:
            filled = sorted([p for p in pool if p.get("id") not in filt["keep_ids"]],
                            key=lambda x: x.get("quality_score", 0), reverse=True)
            for p in filled:
                if len(chosen) >= top_n:
                    break
                chosen.append(p)
        report["gaps"] = filt.get("gaps", [])
        report["missing_queries"] = filt.get("missing_queries", [])

        # ④ 缺口回填：再抓一轮 missing_queries
        mq = [q for q in (filt.get("missing_queries") or []) if isinstance(q, str) and q.strip()]
        if mq and len(chosen) < top_n:
            report["rounds"] = 2
            extra = _fetch_pool(mq[:4], per_query=30, deadline=_deadline)
            extra_ids = {p.get("id") for p in chosen}
            added = [p for p in extra if p.get("id") not in extra_ids]
            pool += added
            # 对新增候选做一次轻量 LM 筛选
            ef = _lm_filter(topic, added) if added else {"keep_ids": None}
            if ef.get("keep_ids") is not None:
                for p in added:
                    if p.get("id") in ef["keep_ids"] and len(chosen) < top_n:
                        r = ef["role_map"].get(p.get("id"))
                        if r:
                            p["role"] = r
                        chosen.append(p)
            report["n_candidates"] = len(pool)
    else:
        # 兜底：规则精选
        chosen = _rules_select(pool, top_n)

    # 最终截到 top_n（无论如何不超上限，且尽量凑满）
    chosen = chosen[:top_n]
    if not chosen and pool:
        chosen = pool[:top_n]

    report["n_selected"] = len(chosen)

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
