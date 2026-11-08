#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反馈台账（Feedback Ledger）——把每次运行的「问题」变成下次运行的约束。

为什么单独做一层：现在的评审/润色/自检各自产出结论，但没有一个地方把它们
攒起来。结果是同一个毛病（比如「结论越界」「文献不足」「方法缺失」）每次都
重新发现、重新解释、重新修，永远学不会。

本模块提供四件事：
  1) 采集：add() 追加一条记录（JSONL，只增不改）；eval / polish / doctor 自动写入
  2) 归类：固定 type 枚举，避免「每次说法都不一样」导致无法统计
  3) 汇总：stats() 给出各类型的出现次数、未修复数、复发率
  4) 消费：lessons() 生成「写作前约束」；export_cases() 生成回归用例清单

记录字段：
  ts / run_dir / topic / stage / type / severity / detail / evidence / action
  / status(open|doing|fixed|wontfix) / source(eval|user|polish|doctor|manual)
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "memory" / "feedback.jsonl"
CASES = ROOT / "memory" / "regression_cases.json"

# 固定类型：新增类型要慎重，否则统计会碎掉
TYPES = {
    "数据错配": "数据集与题目语义不符",
    "方法缺失": "题目点名的方法未执行或被跳过",
    "结构缺陷": "章节/字数/表格/图表缺失或比例失衡",
    "内容矛盾": "结论与数据、显著性、裁决表冲突",
    "文献不足": "文献数量/相关性/覆盖角色不足",
    "写作风格": "套话、单句成段、中英混排、术语不一致",
    "可复现性": "产物缺失、随机种子未固定、路径依赖",
    "其他": "暂未归类的反馈",
}
SEVERITIES = ("P0", "P1", "P2")  # P0=不能交付 P1=明显缺陷 P2=改进项


def _norm_type(t: str) -> str:
    t = (t or "").strip()
    return t if t in TYPES else "其他"


def add(detail: str, type: str = "其他", severity: str = "P2",
        stage: str = "", run_dir: str = "", topic: str = "",
        evidence: str = "", action: str = "", status: str = "open",
        source: str = "manual", provenance: str = "human",
        confidence: str = "medium") -> dict:
    """追加一条反馈记录。返回写入的记录。"""
    rec = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "run_dir": str(run_dir or ""),
        "topic": (topic or "")[:120],
        "stage": stage or "",
        "type": _norm_type(type),
        "severity": severity if severity in SEVERITIES else "P2",
        "detail": (detail or "").strip()[:500],
        "evidence": (evidence or "").strip()[:200],
        "action": (action or "").strip()[:200],
        "status": status if status in ("open", "doing", "fixed", "wontfix") else "open",
        "source": source,
        # 来源与把握程度：参考 nanoMuse 的「记忆要能溯源」——一条规则判出来的问题，
        # 和一个人工批注、模型判断，可信度不同，后续使用时的权重也应不同。
        "provenance": provenance if provenance in ("rule", "model", "human") else "human",
        "confidence": confidence if confidence in ("high", "medium", "low") else "medium",
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def stats() -> dict:
    """按类型统计：总数、未修复、已修复、占比；并给出最该优先处理的类型。"""
    rows = load()
    by_type: dict[str, dict] = {}
    for r in rows:
        t = r.get("type") or "其他"
        d = by_type.setdefault(t, {"total": 0, "open": 0, "fixed": 0})
        d["total"] += 1
        if r.get("status") == "fixed":
            d["fixed"] += 1
        elif r.get("status") != "wontfix":
            d["open"] += 1
    for d in by_type.values():
        d["fix_rate"] = round(d["fixed"] / d["total"], 2) if d["total"] else 0.0
    ranked = sorted(by_type.items(), key=lambda kv: (-kv[1]["open"], -kv[1]["total"]))
    return {"total": len(rows), "by_type": by_type,
            "priority": ranked[0][0] if ranked else None}


def lessons(limit: int = 5, min_count: int = 2) -> list[str]:
    """生成「写作前约束」：反复出现且未修复的问题，转成一句可执行的要求。

    只输出出现 ≥min_count 次且有未修复记录的条目——单次偶发不值得写成规则。
    """
    rows = load()
    agg: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("status") in ("fixed", "wontfix"):
            continue
        agg.setdefault(r.get("type") or "其他", []).append(r)
    _rank = {"human": 3, "model": 2, "rule": 1}
    out = []
    for t, items in sorted(agg.items(), key=lambda kv: -len(kv[1])):
        if len(items) < min_count:
            continue
        items.sort(key=lambda r: _rank.get(r.get("provenance"), 0), reverse=True)
        head = items[0]
        sample = head.get("detail") or ""
        out.append(f"[{t}] 已出现 {len(items)} 次（来源 {head.get('provenance', '?')}），"
                   f"上次：{sample}")
        if len(out) >= limit:
            break
    return out


def mark(index: int, status: str) -> bool:
    """把第 index 条（从 1 开始）标记为 fixed/doing/wontfix。需要重写文件。"""
    rows = load()
    if not (1 <= index <= len(rows)):
        return False
    rows[index - 1]["status"] = status
    LEDGER.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                      encoding="utf-8")
    return True


def _norm_topic(t: str) -> str:
    """题目归一化：去空白、去标点差异、统一大小写，避免同一道题因空格不同而漏配。"""
    s = re.sub(r"[\s　]+", "", (t or ""))
    s = re.sub(r"[，。、；：·（）()\[\]【】]", "", s)
    return s.lower()


def close_for_topic(topic: str, run_dir: str = "", reason: str = "",
                    types: tuple[str, ...] = ()) -> int:
    """同题后续运行过了硬门，就把该题此前的未修复记录销账。

    为什么需要它：`lessons()` 只读 open 记录，而旧实现没有任何路径把 open 变成
    fixed —— 113 条记录、修复率恒为 0，同一批旧账每次写作前都被重新念一遍，
    台账退化成日志。销账条件是「同题 + 后续运行 + 硬门通过」，不是「时间过去了」，
    所以它记录的是真实修复，而不是自动过期。

    只销**别的**运行留下的记录（run_dir 不同），本轮刚记录的问题不会被自己销掉。
    """
    rows = load()
    key = _norm_topic(topic)
    if not key:
        return 0
    here = str(Path(run_dir).resolve()) if run_dir else ""
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    n = 0
    for r in rows:
        if r.get("status") in ("fixed", "wontfix"):
            continue
        if _norm_topic(r.get("topic")) != key:
            continue
        if types and r.get("type") not in types:
            continue
        prev = str(Path(r.get("run_dir")).resolve()) if r.get("run_dir") else ""
        if here and prev == here:
            continue
        r["status"] = "fixed"
        r["closed_ts"] = now
        r["closed_by_run"] = here or str(run_dir or "")
        r["close_reason"] = (reason or "同题后续运行通过硬门")[:200]
        n += 1
    if n:
        LEDGER.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                          encoding="utf-8")
    return n



def export_cases() -> dict:
    """把「已修复且可复现」的问题导出成回归用例清单，防止旧病复发。"""
    rows = [r for r in load() if r.get("status") == "fixed"]
    cases = [{"type": r.get("type"), "detail": r.get("detail"),
              "evidence": r.get("evidence"), "topic": r.get("topic"),
              "checked": False} for r in rows]
    CASES.parent.mkdir(parents=True, exist_ok=True)
    CASES.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"n_cases": len(cases), "path": str(CASES)}


def record_literature_review(review: dict, run_dir: str, topic: str = "") -> int:
    """把文献审评的「缺口/弱相关」落进台账，带上来源与把握程度。

    此前只有评分门的 issue 会入账，文献审评的覆盖缺口每次都随 04 文件一起丢掉。
    模型判断标 model/medium，规则兜底标 rule/medium，人工批注由 --provenance 指定。
    """
    pv = review.get("provenance") if review.get("provenance") in ("rule", "model", "human") else "rule"
    cf = review.get("confidence") if review.get("confidence") in ("high", "medium", "low") else "medium"
    score = review.get("coverage_score")
    n = 0
    for gap in (review.get("missing_themes") or [])[:8]:
        add(str(gap), type="文献不足", severity="P1", stage="literature_review",
            run_dir=run_dir, topic=topic, source="literature_review",
            evidence=f"coverage_score={score}/10；来源={review.get('provenance','?')}",
            action="写入下轮检索式或按该主题补抓文献",
            provenance=pv, confidence=cf)
        n += 1
    weak = review.get("weak_ids") or []
    if weak:
        add(f"审评判定 {len(weak)} 篇相关性弱：{', '.join(map(str, weak[:8]))}",
            type="文献不足", severity="P2", stage="literature_review",
            run_dir=run_dir, topic=topic, source="literature_review",
            evidence=f"coverage_score={score}/10", action="下轮检索降权或剔除",
            provenance=pv, confidence=cf)
        n += 1
    try:
        if score is not None and float(score) < 6:
            add(f"文献覆盖度偏低（{score}/10）：{str(review.get('assessment') or '')[:120]}",
                type="文献不足", severity="P1", stage="literature_review",
                run_dir=run_dir, topic=topic, source="literature_review",
                evidence="04_literature_review.json", action="优先补覆盖而非堆篇数",
                provenance=pv, confidence=cf)
            n += 1
    except (TypeError, ValueError):
        pass
    return n


def record_eval_verdict(verdict: dict, run_dir: str, topic: str = "") -> int:
    """把一次评审结论落进台账：每条 issue 记为一条 P1/P2 反馈（自动采集）。

    来源区分：规则评分门产出的 issue 是 rule/high，LLM 评审官的意见是 model/medium
    ——模型可能误判，权重低于硬规则，也低于人工批注（lessons 里按 human>model>rule 排）。
    """
    n = 0
    for issue in (verdict.get("issues") or [])[:10]:
        t = "其他"
        s = str(issue)
        if any(k in s for k in ("文献", "引用", "参考文献", "外文")):
            t = "文献不足"
        elif any(k in s for k in ("章节", "字数", "篇幅", "下限", "表格", "结构", "字")):
            t = "结构缺陷"
        elif any(k in s for k in ("显著", "矛盾", "结论", "证据")):
            t = "内容矛盾"
        elif any(k in s for k in ("套话", "单句", "段落", "术语")):
            t = "写作风格"
        is_model = s.startswith("LLM评审官") or "幻觉" in s[:20]
        add(s, type=t, severity="P1", stage="eval", run_dir=run_dir,
            topic=topic, source="eval",
            evidence="07_eval_verdict.json",
            provenance=("model" if is_model else "rule"),
            confidence=("medium" if is_model else "high"))
        n += 1
    lj = verdict.get("llm_judge", {}) or {}
    for h in lj.get("hallucinations", [])[:10]:
        add(str(h), type="内容矛盾", severity="P0", stage="eval",
            run_dir=run_dir, topic=topic, source="eval",
            evidence="llm_judge.hallucinations", action="回滚改写，消除虚构数字/事实",
            provenance="model", confidence="medium")
        n += 1
    for c in lj.get("unsupported_claims", [])[:8]:
        add(str(c), type="内容矛盾", severity="P1", stage="eval",
            run_dir=run_dir, topic=topic, source="eval",
            evidence="llm_judge.unsupported_claims", action="写作端下一轮按此改写或删除该断言",
            provenance="model", confidence="medium")
        n += 1
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="反馈台账")
    sub = ap.add_subparsers(dest="cmd")
    a1 = sub.add_parser("add"); a1.add_argument("--detail", required=True)
    a1.add_argument("--type", default="其他"); a1.add_argument("--severity", default="P2")
    a1.add_argument("--stage", default=""); a1.add_argument("--run", default="")
    a1.add_argument("--topic", default=""); a1.add_argument("--source", default="manual")
    sub.add_parser("stats")
    a2 = sub.add_parser("lessons"); a2.add_argument("--min-count", type=int, default=2)
    a3 = sub.add_parser("fix"); a3.add_argument("--index", type=int, required=True)
    a3.add_argument("--status", default="fixed",
                    choices=["fixed", "doing", "wontfix", "open"])
    sub.add_parser("export-cases")
    args = ap.parse_args()
    if args.cmd == "add":
        print(json.dumps(add(args.detail, args.type, args.severity, args.stage,
                             args.run, args.topic, source=args.source),
                         ensure_ascii=False, indent=2))
    elif args.cmd == "stats":
        print(json.dumps(stats(), ensure_ascii=False, indent=2))
    elif args.cmd == "lessons":
        print(json.dumps(lessons(min_count=args.min_count), ensure_ascii=False, indent=2))
    elif args.cmd == "fix":
        print("ok" if mark(args.index, args.status) else "index 越界")
    elif args.cmd == "export-cases":
        print(json.dumps(export_cases(), ensure_ascii=False, indent=2))
    else:
        ap.print_help()
