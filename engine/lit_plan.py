"""检索规划台账：把「拿到选题后判断要抓什么、走哪条路径」这一步留下来。

以前检索式在 GatherAgent 内部生成完就散在日志里，`0x_search_report.json`
只有 queries 列表，没有「为什么是这几条」「走了哪些数据源」「哪几条来自
上一轮反馈」。事后想问「当时为什么没搜这个方向」无从查起。

本模块维护 `<out>/LIT_PLAN.json`，三层信息：
- ``understanding``：选题、任务类型、涉及方法、数据集线索；
- ``strategy``：每条检索式 + 用途分类 + 来源（planner / feedback / refetch /
  broadening）+ 启用的数据源 + 是否由模型规划；
- ``rounds`` / ``outcome``：每一轮实际用了哪些式子、抓回多少、最终入选多少。

用途分类是**确定性词面推断**，不是模型自述，所以字段里明确标了
``purpose_source="keyword_heuristic"``，不冒充模型的推理过程。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

NAME = "LIT_PLAN.json"

_PURPOSE = [
    (("survey", "review", "overview", "综述"), "综述与方法学归纳"),
    (("benchmark", "baseline", "dataset", "corpus", "基准"), "基准数据集与评测"),
    (("application", "case study", "deployment", "实践", "应用"), "应用领域与案例"),
    (("interpretability", "explainab", "evaluation", "评估", "可解释"), "评估与可解释性"),
    (("deep learning", "neural", "transformer", "对比", "compare"), "竞争技术路线"),
]


def classify(query: str) -> str:
    q = str(query or "").lower()
    for keys, label in _PURPOSE:
        if any(k in q for k in keys):
            return label
    return "核心方法与主题主线"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def create(topic: str, queries: list[str], sources: list[str],
           task_type: str = "", methods: list[str] | None = None,
           dataset_query: str = "", lm_driven: bool = False,
           feedback: list[dict] | None = None) -> dict:
    """建立一次检索计划。``feedback`` 是本次从历史评审/审评里取回的检索式。"""
    fb = feedback or []
    fb_queries = {str(x.get("query")) for x in fb}
    groups = []
    for q in queries:
        groups.append({"query": str(q), "purpose": classify(q),
                       "origin": "feedback" if str(q) in fb_queries else "planner"})
    return {
        "topic": topic, "ts": _now(),
        "understanding": {
            "topic": topic, "task_type": task_type,
            "methods": methods or [], "dataset_query": dataset_query,
            "note": "选题解析结果；方法/数据集线索来自 task_spec，不额外推断。",
        },
        "strategy": {
            "lm_driven": lm_driven,
            "sources": sources,
            "planner": "LM 合成" if lm_driven else "关键词派生（网关不可用）",
            "purpose_source": "keyword_heuristic",
            "query_groups": groups,
        },
        "applied_feedback": fb,
        "rounds": [{"round": 1, "kind": "initial", "queries": [str(q) for q in queries],
                    "n_new": 0, "note": "首轮广撒网"}],
        "outcome": {},
        "note": "origin=feedback 的检索式来自上一轮论文诊断/文献审评存入的记忆。",
    }


def add_round(plan: dict, kind: str, queries: list[str], n_new: int = 0,
              note: str = "") -> dict:
    plan.setdefault("rounds", []).append({
        "round": len(plan["rounds"]) + 1, "kind": kind,
        "queries": [str(q) for q in queries], "n_new": n_new, "note": note})
    for q in queries:
        plan["strategy"]["query_groups"].append({
            "query": str(q), "purpose": classify(q),
            "origin": "refetch" if kind in ("refetch", "gap") else "broadening"})
    return plan


def finish(plan: dict, n_candidates: int, n_selected: int,
           gaps: list[str] | None = None) -> dict:
    plan["outcome"] = {"n_candidates": n_candidates, "n_selected": n_selected,
                       "gaps": gaps or [], "finished_at": _now()}
    return plan


def write(out_dir, plan: dict) -> str:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    f = p / NAME
    f.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(f)


def load(out_dir) -> dict | None:
    f = Path(out_dir) / NAME
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
