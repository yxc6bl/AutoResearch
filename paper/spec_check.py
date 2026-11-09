"""素材完备度前置检查：落笔之前先算一遍「素材够不够撑起这篇论文」。

为什么需要这一步
----------------
系统执行零编造：模型只能复述 task_spec.json 里给出的素材。素材少一个字段，
正文就少一块实质内容，只能靠「具有重要意义」「为后续研究提供参考」这类空话
填满字数，评审官会照样判「缺乏依据」。等到写完再被评审指出，已经白花了
几分钟的模型调用。

所以把缺口检查放到最前面：缺什么、建议补什么，一次列清楚。检查只提示、
不阻断——素材再薄也仍然产出一份结构完整的稿子，但会在
``00_spec_check.json`` 里留痕，说明这篇是在什么素材密度下写的。

用法
----
    python paper/spec_check.py --spec out_run/task_spec.json
    python paper/spec_check.py --spec task_spec.json --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 各类型的字段要求 ───────────────────────────────────────────────────
# (字段路径, 人类可读名称, 类型, 阈值, 是否必需)
#   text           : 字符串，阈值是最少字数
#   list           : 列表，阈值是最少条数
#   filled_items   : a[].b 形式，阈值是「至少多少条有值」
REQUIREMENTS: dict[str, list[tuple[str, str, str, int, bool]]] = {
    "survey": [
        ("title", "论文题目", "text", 6, True),
        ("literature_query", "检索式（英文关键词）", "text", 8, True),
    ],
    "system_design": [
        ("title", "论文题目", "text", 6, True),
        ("system_spec", "系统设计说明（system_spec）", "any", 1, True),
        ("system_spec.summary", "系统一句话概述", "text", 40, True),
        ("system_spec.tech_stack", "技术栈", "list", 3, True),
        ("system_spec.requirements.functional", "功能性需求", "list", 3, True),
        ("system_spec.requirements.non_functional", "非功能性需求", "list", 2, False),
        ("system_spec.architecture.layers", "架构分层", "list", 2, True),
        ("system_spec.modules", "模块清单", "list", 3, True),
        ("system_spec.modules[].responsibility", "各模块职责", "filled_items", 3, True),
        ("system_spec.validation", "验证与测试安排", "any", 1, True),
        ("system_spec.validation.test_cases", "可核验的测试用例", "list", 2, False),
    ],
    "case_study": [
        ("title", "论文题目", "text", 6, True),
        ("case_spec", "案例材料（case_spec）", "any", 1, True),
        ("case_spec.subject", "案例对象", "text", 4, True),
        ("case_spec.background", "案例背景", "text", 40, True),
        ("case_spec.phenomenon", "要解释的现象或问题", "text", 20, True),
        ("case_spec.framework", "分析框架", "text", 6, True),
        ("case_spec.materials", "资料清单", "list", 2, True),
        ("case_spec.analysis_points", "分析维度", "list", 2, False),
    ],
    "technical_report": [
        ("title", "论文题目", "text", 6, True),
        ("project_spec", "项目材料（project_spec）", "any", 1, True),
        ("project_spec.background", "项目背景", "text", 30, True),
        ("project_spec.objectives", "建设目标", "list", 2, True),
        ("project_spec.scope", "建设范围（含不做什么）", "text", 15, True),
        ("project_spec.work_items", "实施工作项", "list", 3, True),
        ("project_spec.deliverables", "交付物", "list", 2, True),
        ("project_spec.acceptance", "验收标准", "list", 2, True),
        ("project_spec.issues", "实施中遇到的问题", "list", 1, False),
    ],
    "research_proposal": [
        ("title", "论文题目", "text", 6, True),
        ("proposal_spec", "研究方案（proposal_spec）", "any", 1, True),
        ("proposal_spec.background", "研究背景", "text", 30, True),
        ("proposal_spec.questions", "研究问题", "list", 2, True),
        ("proposal_spec.content", "研究内容", "list", 2, True),
        ("proposal_spec.methods", "研究方法", "list", 2, True),
        ("proposal_spec.schedule", "进度安排", "list", 2, True),
        ("proposal_spec.feasibility", "可行性说明", "text", 10, False),
    ],
}

# 每种类型「最该补」的那一块，用于给出一句方向性建议
_CRITICAL_BLOCK = {
    "system_design": ("system_spec", "把技术栈、功能需求、架构分层、模块职责写全，"
                                    "模型才能写出一篇具体的设计文，而不是通用套话"),
    "case_study": ("case_spec", "补上案例对象、背景、现象、分析框架与资料清单，"
                                "案例研究的分量全在这块素材上"),
    "technical_report": ("project_spec", "补上目标、范围、工作项、交付物与验收标准，"
                                         "工程报告的说服力来自可核验的过程"),
    "research_proposal": ("proposal_spec", "补上研究问题、内容、方法与进度，"
                                           "开题报告的评审重点就是这几项"),
}


def _get(spec: dict, path: str):
    """按路径取值；支持 a.b.c 与 a[].b（返回列表，元素为各条目的取值）。"""
    if "[]" in path:
        base, sub = path.split("[]", 1)
        sub = sub.lstrip(".")
        node = _get(spec, base)
        if not isinstance(node, list):
            return None
        if not sub:
            return node
        return [_get(x, sub) for x in node if isinstance(x, dict)]
    cur = spec
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _judge(value, kind: str, threshold: int) -> tuple[str, str]:
    """返回 (status, 说明)。status ∈ ok / missing / thin。"""
    if kind == "text":
        if not isinstance(value, str) or not value.strip():
            return "missing", "未填写"
        n = len(value.strip())
        return ("ok", f"{n} 字") if n >= threshold else ("thin", f"仅 {n} 字（建议 ≥{threshold}）")
    if kind == "list":
        if not isinstance(value, list) or not value:
            return "missing", "未填写"
        return (("ok", f"{len(value)} 条") if len(value) >= threshold
                else ("thin", f"仅 {len(value)} 条（建议 ≥{threshold}）"))
    if kind == "filled_items":
        vals = [v for v in (value or []) if isinstance(v, str) and v.strip()]
        if not vals:
            return "missing", "未填写"
        return (("ok", f"{len(vals)} 条有值") if len(vals) >= threshold
                else ("thin", f"仅 {len(vals)} 条有值（建议 ≥{threshold}）"))
    # any
    if value in (None, "", [], {}):
        return "missing", "未填写"
    return "ok", "已提供"


def check_spec(paper_type: str, spec: dict) -> dict:
    """检查一份 spec 对某论文类型的素材完备度。"""
    reqs = REQUIREMENTS.get(paper_type, [])
    items, missing, thin = [], [], []
    got = 0.0
    for path, label, kind, thr, required in reqs:
        status, detail = _judge(_get(spec, path), kind, thr)
        # 计分：必需项各 1 分，非必需项 0.5 分；missing 记 0，thin 记半分
        weight = 1.0 if required else 0.5
        if status == "ok":
            got += weight
        elif status == "thin":
            got += weight * 0.5
        items.append({"field": path, "label": label, "status": status,
                      "detail": detail, "required": required})
        if status == "missing":
            (missing if required else thin).append(label)
        elif status == "thin":
            thin.append(label)
    total = sum(1.0 if r[4] else 0.5 for r in reqs) or 1.0
    score = round(100 * got / total)
    block, hint = _CRITICAL_BLOCK.get(paper_type, ("", ""))
    block_missing = bool(block) and _judge(_get(spec, block), "any", 1)[0] == "missing"
    if block_missing:
        hint = f"最关键的一块素材（{block}）没提供：{hint}。"
    elif missing:
        hint = "缺的是：" + "、".join(missing) + "；补齐后重跑，正文的实质内容会明显变厚。"
    elif thin:
        hint = "主要字段都在，但偏薄（" + "、".join(thin[:4]) + "），可再补细节。"
    else:
        hint = "素材齐备。"
    return {
        "paper_type": paper_type, "score": score,
        "ok": not missing, "items": items,
        "missing": missing, "thin": thin, "hint": hint,
    }


def format_report(sc: dict) -> list[str]:
    """把检查结果变成几行人能读的提示。"""
    if not sc or not sc.get("items"):
        return []
    tag = "齐备" if sc.get("ok") else "有缺口"
    out = [f"[素材] 完备度 {sc['score']}/100（{tag}）：{sc.get('hint', '')}"]
    if not sc.get("ok"):
        for it in sc.get("items", []):
            if it["status"] == "missing" and it["required"]:
                out.append(f"[素材]   缺 {it['field']}（{it['label']}）")
    for it in sc.get("items", []):
        if it["status"] == "thin":
            out.append(f"[素材]   薄 {it['field']}（{it['label']}）：{it['detail']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="论文素材完备度检查")
    ap.add_argument("--spec", required=True, help="task_spec.json 路径")
    ap.add_argument("--paper-type", default="", help="覆盖 spec 里的 paper_type")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    pt = args.paper_type or spec.get("paper_type") or "method_comparison"
    sc = check_spec(pt, spec)
    if args.json:
        print(json.dumps(sc, ensure_ascii=False, indent=2))
    else:
        print("\n".join(format_report(sc)))
    return 0 if sc["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
