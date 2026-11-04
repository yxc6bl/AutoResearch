#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闸门台账（Gate Ledger）——把散落在各处的「检查点」收成一张可审计的表。

为什么要这层：系统里早就有硬门（零编造）、降级判定（DELIVERY_STATUS）、评分门
（EvalAgent）、文献覆盖审评，但它们分散在四个文件里，谁也说不清「这次到底卡在哪、
谁能放行」。参考系统论文里这叫权限/闸门表（Sentinel 那类东西），它同时是工程文档
和架构论证的证据。

本模块**不新增限制**：CATALOG 只登记代码里已经存在的判定，状态一律从真实产物读取，
读不到就是 unknown（未验证），绝不猜测成 pass 或 fail。分级只影响报告与交付裁决：

  P0 不可交付   —— 严格模式下 fail 或 unknown 都阻塞（写死质量底线，不迁就网关）
  P1 明显缺陷   —— 提示，不阻塞；人工可留痕放行
  P2 信息性     —— 只记录，用于复盘与写系统实现章

人工放行：approve() 写 07_gate_approvals.json，必须带 who/why/ts，报告里标 waived，
不抹掉原始 fail 证据。放行是「人做了决定」，不是「系统把问题改好了」。

产物：
  <out_dir>/07_gate_approvals.json   人工放行记录（只增不改）
  <out_dir>/07_gate_report.json     本次裁决报告（每次刷新覆盖）
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

REPORT_FILE = "07_gate_report.json"
APPROVALS_FILE = "07_gate_approvals.json"
SCHEMA = "gate-report/v1"

# 闸门清单：新增一条闸门必须同时给出产物路径和判定依据，否则不许进表。
# authority 写清楚谁能放行，这是「权限边界」而不是一句口号。
CATALOG = [
    {"id": "G01", "level": "P0", "stage": "experiment", "name": "分析层产出有效",
     "artifact": "03_experiment.json", "rule": "无 error，且有 metrics 与 best_method",
     "action": "实验失败不得成文，先修数据/方法绑定", "authority": "系统拒绝；人工放行需说明数据来源"},
    {"id": "G02", "level": "P0", "stage": "write", "name": "正文存在且篇幅达标",
     "artifact": "06_paper/paper_full.md", "rule": "正文汉字数 ≥ 6000（与 mark_delivery 同口径）",
     "action": "模板骨架稿不可交付，等模型恢复后重跑写作", "authority": "系统拒绝；人工放行需书面确认"},
    {"id": "G03", "level": "P0", "stage": "write", "name": "模型参与写作",
     "artifact": "06_paper/DELIVERY_STATUS.json", "rule": "deliverable=true 且 llm_written 不为 False",
     "action": "确定性降级稿仅用于自测/定位链路", "authority": "系统拒绝；人工放行需书面确认"},
    {"id": "G04", "level": "P0", "stage": "eval", "name": "零编造硬门",
     "artifact": "06_paper/07_eval_verdict.json",
     "rule": "LLM 评审官已介入且未发现幻觉；gate=hard_pass、consistency=100",
     "action": "结论与裁决表矛盾/编造数字，必须回滚改写", "authority": "系统拒绝，不接受放行"},
    {"id": "G05", "level": "P1", "stage": "eval", "name": "质量评分门通过",
     "artifact": "06_paper/07_eval_verdict.json", "rule": "passed=true 且 overall ≥ 60",
     "action": "按 issues/suggestions 补强后重评", "authority": "导师或作者可留痕放行"},
    {"id": "G06", "level": "P1", "stage": "literature", "name": "文献池非空",
     "artifact": "01_literature.json", "rule": "papers 数量 > 0（不设固定篇数下限）",
     "action": "补抓文献，篇数由题目需求与角色覆盖决定", "authority": "作者可留痕放行"},
    {"id": "G07", "level": "P2", "stage": "literature", "name": "文献理解可溯源",
     "artifact": "01_literature.json", "rule": "记录带 llm_what / llm_fit 的文献比例（只报告，不判死）",
     "action": "覆盖率低说明本轮只做了关键词筛选，需在报告里如实写明", "authority": "无需放行"},
    {"id": "G08", "level": "P2", "stage": "literature", "name": "文献审评已执行",
     "artifact": "04_literature_review.json", "rule": "记录覆盖度、是否 LM 驱动、缺口主题",
     "action": "缺口主题写入下轮检索式", "authority": "无需放行"},
    {"id": "G09", "level": "P2", "stage": "write", "name": "文献真正用进正文",
     "artifact": "05_lit_feedback.json", "rule": "n_cited > 0（入选文献被正文实际引用）",
     "action": "检查第 2 章引用块是否落到正文", "authority": "无需放行"},
    {"id": "G10", "level": "P2", "stage": "eval", "name": "LLM 评审官是否介入",
     "artifact": "06_paper/07_eval_verdict.json", "rule": "llm_judge.available=true",
     "action": "网关不可用时只有规则评分，报告中必须标明评审覆盖面不足", "authority": "无需放行"},
    {"id": "G11", "level": "P2", "stage": "trace", "name": "执行轨迹可审计",
     "artifact": "00_trace.jsonl", "rule": "有阶段记录且无 error 状态",
     "action": "看 trace 定位失败阶段与耗时", "authority": "无需放行"},
]

_BY_ID = {g["id"]: g for g in CATALOG}


def _json(out: Path, rel: str):
    p = out / rel
    if not p.exists():
        return None, False
    try:
        return json.loads(p.read_text(encoding="utf-8")), True
    except Exception:
        return None, True  # 文件在但坏了，不能当「未跑」


def _cn_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def _row(meta: dict, status: str, evidence: str,
         provenance: str = "rule", confidence: str = "high") -> dict:
    return {**meta, "status": status, "evidence": evidence[:300],
            "provenance": provenance, "confidence": confidence}


def _evaluate_one(out: Path, meta: dict) -> dict:
    gid = meta["id"]
    unknown = lambda ev: _row(meta, "unknown", ev, "rule", "high")  # noqa: E731
    pass_ = lambda ev, pv="rule", cf="high": _row(meta, "pass", ev, pv, cf)  # noqa: E731
    fail = lambda ev, pv="rule", cf="high": _row(meta, "fail", ev, pv, cf)  # noqa: E731

    if gid == "G01":
        d, exists = _json(out, meta["artifact"])
        if not exists:
            return unknown("未找到 03_experiment.json（本次未跑实验）")
        if d is None:
            return fail("03_experiment.json 无法解析")
        if isinstance(d, dict) and d.get("error"):
            return fail(f"分析层报错：{str(d['error'])[:120]}")
        analysis = d.get("analysis", d) if isinstance(d, dict) else {}
        n = len(analysis.get("metrics") or [])
        best = analysis.get("best_method")
        if not n or not best:
            return fail(f"metrics={n}，best_method={best}")
        return pass_(f"metrics={n}，best_method={best}={analysis.get('best_value')}")

    if gid == "G02":
        p = out / meta["artifact"]
        if not p.exists():
            return unknown("未找到 06_paper/paper_full.md（本次未出稿）")
        n = _cn_chars(p.read_text(encoding="utf-8", errors="ignore"))
        return (pass_(f"正文 {n} 个汉字") if n >= 6000
                else fail(f"正文仅 {n} 个汉字（门槛 6000，与 mark_delivery 同口径）"))

    if gid == "G03":
        d, exists = _json(out, meta["artifact"])
        if not exists:
            return unknown("未写 DELIVERY_STATUS.json；无法判定模型是否参与写作")
        if d is None:
            return fail("DELIVERY_STATUS.json 无法解析")
        if d.get("deliverable") is False:
            return fail("已标不可交付：" + "；".join(d.get("reasons") or []))
        if d.get("llm_written") is False:
            return fail("llm_written=False（走了确定性模板骨架）")
        if d.get("llm_written") is None:
            return unknown("DELIVERY_STATUS 未记录 llm_written，模型参与情况未验证")
        return pass_(f"deliverable=true，llm_written={d.get('llm_written')}，"
                     f"汉字 {d.get('chinese_chars')}")

    if gid in ("G04", "G05", "G10"):
        d, exists = _json(out, meta["artifact"])
        if not exists:
            return unknown("未找到 07_eval_verdict.json（本次未跑评分门）")
        if d is None:
            return fail("07_eval_verdict.json 无法解析")
        lj = d.get("llm_judge") or {}
        if gid == "G04":
            hall = lj.get("hallucinations") or []
            consistency = d.get("scores", {}).get("consistency")
            # 双保险：既看评审官原始幻觉清单，也看评分门最终有没有把幻觉纳入硬门。
            # 老版本 eval_agent 会覆盖 gate（实测 bug），只靠 gate 会漏判。
            lj_bad = bool(d.get("llm_hallucination_gate")) or bool(hall)
            # 评审官缺席 ≠ 没有幻觉。语义编造只有它抓得到，缺席时记「未验证」，
            # 严格模式下同样阻塞；否则网关一挂，幻觉稿就会被当成零编造合格稿。
            if not lj.get("available") and not hall:
                return unknown("LLM 评审官未介入（" +
                               str(lj.get("reason") or "verdict 无记录") +
                               "）：语义幻觉未验证，不能判零编造硬门通过")
            if d.get("gate") != "hard_pass" or consistency == 0 or lj_bad:
                _h = "；".join(str(x)[:80] for x in hall[:2])
                return fail(f"gate={d.get('gate')}，consistency={consistency}，"
                            f"幻觉 {len(hall)} 处" + (f"：{_h}" if _h else ""),
                            ("model" if hall else "rule"),
                            ("medium" if hall else "high"))
            return pass_(f"gate=hard_pass，consistency={consistency}，无幻觉")
        if gid == "G05":
            if not d.get("passed") or float(d.get("overall") or 0) < 60:
                return fail(f"passed={d.get('passed')}，overall={d.get('overall')}")
            return pass_(f"passed=true，overall={d.get('overall')}，等级 {d.get('grade')}")
        if lj.get("available"):
            return pass_(f"LLM 评审介入，评分 {lj.get('overall')}，结论 {lj.get('verdict')}",
                         "model", "medium")
        return fail("llm_judge.available≠true（" + str(lj.get("reason") or "未介入") +
                    "）：本轮只有规则评分，G04 因此记为未验证而非通过")

    if gid == "G06":
        d, exists = _json(out, meta["artifact"])
        if not exists:
            return unknown("未找到 01_literature.json（本次未抓文献）")
        papers = (d or {}).get("papers") or [] if isinstance(d, dict) else []
        if not papers:
            return fail("papers 为空")
        return pass_(f"入选 {len(papers)} 篇（不设固定篇数门槛）")

    if gid == "G07":
        d, exists = _json(out, meta["artifact"])
        if not exists:
            return unknown("未找到 01_literature.json")
        papers = (d or {}).get("papers") or [] if isinstance(d, dict) else []
        if not papers:
            return unknown("无文献，覆盖率无法计算")
        n_read = sum(1 for p in papers if p.get("confidence") == "lm_read"
                     or p.get("llm_what") or p.get("llm_fit") is not None)
        n_use = sum(1 for p in papers if p.get("llm_usable_in")
                    and p.get("llm_usable_in") != "无")
        tiers: dict[str, int] = {}
        for p in papers:
            k = p.get("confidence") or "未标注"
            tiers[k] = tiers.get(k, 0) + 1
        detail = "、".join(f"{k}={v}" for k, v in sorted(tiers.items()))
        pct = round(100 * n_read / len(papers), 1)
        if n_read == 0:
            return fail(f"0/{len(papers)} 篇带 LM 阅读判断：本轮纯关键词兜底，"
                        f"未做「读摘要判断能不能用」，文献质量只能算低可信")
        return pass_(f"{n_read}/{len(papers)} 篇带 LM 阅读判断（{pct}%），"
                     f"其中 {n_use} 篇标明用在哪节；可信度分布：{detail}")

    if gid == "G08":
        d, exists = _json(out, meta["artifact"])
        if not exists:
            return unknown("未写 04_literature_review.json（本轮复用已有文献，未重新审评）")
        if d is None:
            return fail("04_literature_review.json 无法解析")
        return pass_(f"覆盖度 {d.get('coverage_score')}/10，"
                     f"{'LM 审评' if d.get('lm_driven') else '规则兜底'}，"
                     f"缺口 {len(d.get('missing_themes') or [])} 项")

    if gid == "G09":
        d, exists = _json(out, meta["artifact"])
        if not exists:
            return unknown("未写 05_lit_feedback.json（本次未生成使用反馈）")
        n = (d or {}).get("n_cited", 0) if isinstance(d, dict) else 0
        return (pass_(f"正文实际引用 {n} 篇 / 入选 {(d or {}).get('n_selected')} 篇")
                if n else fail("n_cited=0：入选文献一篇都没进正文"))

    if gid == "G11":
        p = out / meta["artifact"]
        if not p.exists():
            return unknown("未找到 00_trace.jsonl（旧运行或 step 单步前）")
        rows = []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        if not rows:
            return unknown("00_trace.jsonl 为空")
        bad = [r.get("stage") for r in rows if r.get("status") != "ok"]
        return (fail(f"{len(rows)} 条阶段记录，失败阶段：{bad}") if bad
                else pass_(f"{len(rows)} 条阶段记录，无失败阶段"))

    return unknown("未实现的闸门（请检查 CATALOG 与 _evaluate_one 是否同步）")


def load_approvals(out: Path) -> list[dict]:
    d, exists = _json(out, APPROVALS_FILE)
    if not exists:
        return []
    return d if isinstance(d, list) else []


def approve(out: Path, gate_ids: list[str], who: str, why: str) -> dict:
    """人工放行：必须写明谁、为什么。P0 也允许，但会永久留痕且不抹掉原始证据。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_approvals(out)
    known = [g for g in gate_ids if g in _BY_ID]
    unknown = [g for g in gate_ids if g not in _BY_ID]
    for gid in known:
        rows.append({
            "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "gate_id": gid, "level": _BY_ID[gid]["level"], "who": who, "why": why,
            "provenance": "human", "confidence": "high", "scope": "this_run",
        })
    (out / APPROVALS_FILE).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"approved": known, "unknown_gate_ids": unknown, "n_records": len(rows)}


def report(out, strict: bool = True, write: bool = True) -> dict:
    """从真实产物生成闸门报告。strict 控制 P0 的 unknown 是否阻塞交付。"""
    out = Path(out)
    approvals = load_approvals(out)
    waived = {a.get("gate_id") for a in approvals}
    rows = []
    for meta in CATALOG:
        r = _evaluate_one(out, meta)
        if r["id"] in waived:
            a = [x for x in approvals if x.get("gate_id") == r["id"]][-1]
            r["status"] = "waived"
            r["provenance"] = "human"
            r["confidence"] = "high"
            r["waived_by"] = a.get("who", "")
            r["waived_why"] = a.get("why", "")
            r["raw_status"] = r["evidence"]  # 原始证据保留，不因放行消失
        rows.append(r)

    counts: dict[str, dict] = {}
    for lv in ("P0", "P1", "P2"):
        sub = [r for r in rows if r["level"] == lv]
        counts[lv] = {s: sum(1 for r in sub if r["status"] == s)
                      for s in ("pass", "fail", "unknown", "waived")}
        counts[lv]["total"] = len(sub)
    blocking = [r["id"] for r in rows if r["level"] == "P0" and r["status"] == "fail"]
    unverified = [r["id"] for r in rows if r["level"] == "P0" and r["status"] == "unknown"]
    deliverable = not blocking and not (unverified if strict else [])
    rep = {
        "schema": SCHEMA,
        "out_dir": str(out),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "strict": bool(strict),
        "counts": counts,
        "blocking": blocking,
        "unverified": unverified,
        "approvals": approvals,
        "decision": {
            "deliverable": bool(deliverable),
            "basis": ("P0 全部通过或已留痕放行" if deliverable else
                      "阻塞：" + "、".join(blocking) +
                      ("；未验证：" + "、".join(unverified) if (strict and unverified) else "")),
            "human_override": [a["gate_id"] for a in approvals
                               if a.get("level") == "P0"],
        },
        "gates": rows,
    }
    if write:
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / REPORT_FILE).write_text(
                json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return rep


def sync_delivery(out, rep: dict | None = None) -> dict | None:
    """让 DELIVERY_STATUS.json 与闸门裁决一致，消除「可交付 / 不可交付」两种结论。

    mark_delivery 只看字数与模型参与，看不到评审幻觉；闸门层裁决后要把结论回写。
    也可以对历史产物直接调用一次重算（rep 不传就现场生成）。
    """
    out = Path(out)
    p = out / "06_paper" / "DELIVERY_STATUS.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    rep = rep or report(out, strict=True, write=False)
    dec = rep["decision"]
    d["gate_decision"] = {
        "deliverable": dec["deliverable"],
        "blocking": rep["blocking"], "unverified": rep["unverified"],
        "report": str(out / REPORT_FILE),
    }
    if not dec["deliverable"] and d.get("deliverable"):
        d["deliverable"] = False
        why = rep["blocking"] or rep["unverified"]
        d["reasons"] = (d.get("reasons") or []) + [
            "P0 闸门未通过/未验证：" + "、".join(why)]
        d["hint"] = "不可交付：以 07_gate_report.json 的 P0 裁决为准"
    try:
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return d


def refresh(out, strict: bool = True) -> dict:
    """静默刷新（流水线内部调用，不打印）。"""
    return report(out, strict=strict, write=True)


_MARK = {"pass": "通过", "fail": "未通过", "unknown": "未验证", "waived": "已放行"}


def render(rep: dict) -> str:
    lines = [f"=== 闸门报告 {rep['out_dir']} ==="]
    for lv, label in (("P0", "不可交付"), ("P1", "明显缺陷"), ("P2", "信息性")):
        c = rep["counts"][lv]
        lines.append(f"{lv} {label}：共 {c['total']} 项｜通过 {c['pass']}｜"
                     f"未通过 {c['fail']}｜未验证 {c['unknown']}｜已放行 {c['waived']}")
    lines.append("")
    for r in rep["gates"]:
        lines.append(f"  [{r['level']}][{_MARK.get(r['status'], r['status'])}] "
                     f"{r['id']} {r['name']} — {r['evidence']}")
        if r["status"] in ("fail", "unknown") and r["level"] == "P0":
            lines.append(f"      处置：{r['action']}｜放行权限：{r['authority']}")
        if r["status"] == "waived":
            lines.append(f"      人工放行：{r.get('waived_by')}（{r.get('waived_why')}）")
    lines.append("")
    d = rep["decision"]
    lines.append(("结论：可交付" if d["deliverable"] else "结论：不可交付") +
                 f"（严格模式={'开' if rep['strict'] else '关'}）—— {d['basis']}")
    if d["human_override"]:
        lines.append(f"注意：存在 P0 人工放行 {d['human_override']}，已留痕，未改写原始证据。")
    lines.append(f"报告：{Path(rep['out_dir']) / REPORT_FILE}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="闸门台账：分级检查、交付裁决、人工放行留痕")
    ap.add_argument("--out", required=True, help="产物目录")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    ap.add_argument("--no-strict", action="store_true", help="P0 未验证不阻塞（仅查看）")
    ap.add_argument("--approve", default="", help="人工放行的闸门 id，逗号分隔，如 G03,G05")
    ap.add_argument("--who", default="", help="放行人（必填）")
    ap.add_argument("--why", default="", help="放行理由（必填）")
    a = ap.parse_args()
    if a.approve:
        if not a.who or not a.why:
            raise SystemExit("人工放行必须同时给 --who 与 --why")
        print(json.dumps(approve(Path(a.out), [x.strip() for x in a.approve.split(",")],
                                 a.who, a.why), ensure_ascii=False, indent=2))
    r = report(Path(a.out), strict=not a.no_strict)
    print(json.dumps(r, ensure_ascii=False, indent=2) if a.json else render(r))
