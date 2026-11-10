"""选材台账 + 人工偏好记忆：把「抓了哪些 / 谁选了哪些 / 为什么」留下来，并让下次真的改进。

三个层次，别混在一起
--------------------
1. **本次台账** ``<out>/LIT_SELECTION.json``：候选池里每一篇的去留、理由和决策者
   （model / rule / human）。以前只有 02 候选池和 01 入选集两个快照，中间那一层
   「为什么选它、谁选的」完全没留痕，用户想改也无从下手。
2. **跨运行偏好** ``memory/lit_preferences.json``：人对文献的取舍与规则
   （「别用预印本」「优先近三年」「这篇别选」）。单条意见立刻生效，不必等
   ``feedback.lessons()`` 攒够两次。
3. **评审批注** 同一份偏好文件里：人对诊断报告某条意见的采纳/驳回，下一轮
   写作提示会读到它。

应用时机：抓完、写正文之前。只做「剔除」与「加权」，**不重排入选顺序**——
顺序决定正文 [N] 引用编号，重排会把引用全打乱。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREF_PATH = ROOT / "memory" / "lit_preferences.json"
LEDGER_NAME = "LIT_SELECTION.json"

_KINDS = {"avoid", "require", "boost"}
_FIELDS = {"id", "title", "venue", "year", "keyword", "topic"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(s or "").lower())[:120]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ── 偏好存储 ──────────────────────────────────────────────────────────
def _empty() -> dict:
    return {"version": 2, "rules": [], "paper_decisions": [],
            "review_notes": [], "feedback_queries": []}


def load_prefs() -> dict:
    if not PREF_PATH.exists():
        return _empty()
    try:
        d = json.loads(PREF_PATH.read_text(encoding="utf-8"))
        for k, v in _empty().items():
            d.setdefault(k, v)
        return d
    except Exception:
        return _empty()


def _save(d: dict) -> None:
    PREF_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREF_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def add_rule(kind: str, field: str, value: str, note: str = "",
             scope: str = "global", weight: float = 0.15,
             source: str = "human") -> dict:
    """新增一条偏好规则。

    kind   : avoid=命中即剔除 / require=命中优先 / boost=命中加分
    field  : id / title / venue / year / keyword / topic
    value  : 匹配值；year 支持 ">=2022"、"<=2020"
    scope  : global=所有题目；其它字符串视为题目关键词，只对匹配的题目生效
    """
    if kind not in _KINDS or field not in _FIELDS:
        raise ValueError(f"kind 必须 ∈ {_KINDS}，field 必须 ∈ {_FIELDS}")
    rec = {"id": f"r{len(load_prefs()['rules']) + 1}", "kind": kind,
           "field": field, "value": str(value), "scope": scope or "global",
           "weight": float(weight), "note": note, "source": source,
           "ts": _now(), "active": True}
    d = load_prefs()
    d["rules"].append(rec)
    _save(d)
    return rec


def add_decision(key: str, action: str, note: str = "", topic: str = "") -> dict:
    """对具体某篇文献表态：key 可以是 id、DOI 或题名（归一化后匹配）。"""
    if action not in ("keep", "drop"):
        raise ValueError("action 只能是 keep 或 drop")
    rec = {"key": _norm(key), "raw": key, "action": action, "note": note,
           "topic": topic, "ts": _now(), "source": "human"}
    d = load_prefs()
    d["paper_decisions"] = [x for x in d["paper_decisions"]
                            if not (x.get("key") == rec["key"] and x.get("action") == action)]
    d["paper_decisions"].append(rec)
    _save(d)
    return rec


def add_review_note(run_dir: str, item_id: str, status: str,
                    note: str = "", detail: str = "") -> dict:
    """人对诊断报告某条意见的批注：accepted / rejected / done。"""
    if status not in ("accepted", "rejected", "done"):
        raise ValueError("status 只能是 accepted / rejected / done")
    rec = {"run_dir": str(run_dir), "item_id": item_id, "status": status,
           "note": note, "detail": detail, "ts": _now(), "source": "human"}
    d = load_prefs()
    d["review_notes"].append(rec)
    _save(d)
    return rec


# ── 评审反馈 → 下轮检索式（系统自我改进） ─────────────────────────────
def _paper_hits(keys: set[str], p: dict) -> bool:
    """人工表态是否命中这篇文献。

    短标识（如测试里的 a/b，或用户手输的简写）绝不做子串匹配——
    "a" 会出现在几乎任何英文题名里，实测把「人工剔除」误豁免成「保留」。
    只有完整 id / 题名精确相等，或足够长的标识（≥8 字符）才允许子串命中。
    """
    ident = _norm(f"{p.get('id') or ''}{p.get('title') or ''}")
    pid = _norm(p.get("id"))
    ptitle = _norm(p.get("title"))
    for k in keys or ():
        if not k:
            continue
        if k == pid or k == ptitle or (len(k) >= 8 and k in ident):
            return True
    return False


def add_feedback_queries(queries: list[str], source: str,
                         note: str = "") -> int:
    """把论文诊断 / 文献审评指出的缺口转成下轮要搜的式子，存进记忆。

    去重按归一化文本；同一条反馈只存一次。source 记来源，便于事后查
    「这条检索式是为什么加的」。
    """
    d = load_prefs()
    d.setdefault("feedback_queries", [])
    seen = {_norm(x.get("query")) for x in d["feedback_queries"]}
    n = 0
    for q in queries or []:
        q = str(q).strip()
        if not q or _norm(q) in seen:
            continue
        d["feedback_queries"].append({
            "query": q[:200], "source": source, "note": note,
            "ts": _now(), "used_at": None})
        seen.add(_norm(q))
        n += 1
    if n:
        _save(d)
    return n


def take_feedback_queries(limit: int = 4) -> list[dict]:
    """取出尚未消费的反馈检索式，并标记已被本次检索计划采用。"""
    d = load_prefs()
    out = []
    for x in d.setdefault("feedback_queries", []):
        if x.get("used_at"):
            continue
        out.append({"query": x["query"], "source": x.get("source"),
                    "note": x.get("note"), "ts": x.get("ts")})
        x["used_at"] = _now()
        if len(out) >= limit:
            break
    if out:
        _save(d)
    return out


def apply_human_choices(run_dir, keep: list[str], drop: list[str],
                        note: str = "") -> dict:
    """网页/命令行里人勾选的保留与剔除，一次写回偏好台账与本次选材台账。"""
    keep, drop = list(keep or []), list(drop or [])
    for k in keep:
        add_decision(k, "keep", note)
    for k in drop:
        add_decision(k, "drop", note)
    p = Path(run_dir) / LEDGER_NAME
    updated = 0
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            dk = {_norm(k) for k in drop}
            kk = {_norm(k) for k in keep}
            for row in d.get("decisions") or []:
                ident = _norm(f"{row.get('id') or ''}{row.get('title') or ''}")
                if _paper_hits(dk, row):
                    row["by"] = "human"
                    row["selected"] = False
                    row["reason"] = note or "人工剔除"
                    updated += 1
                elif _paper_hits(kk, row):
                    row["by"] = "human"
                    row["selected"] = True
                    row["reason"] = note or "人工要求保留"
                    updated += 1
            d["human_choices"] = {"keep": keep, "drop": drop,
                                  "note": note, "ts": _now()}
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return {"keep": len(keep), "drop": len(drop),
            "ledger_updated": updated, "path": str(p)}


def active_rules(topic: str = "") -> list[dict]:
    d = load_prefs()
    t = _norm(topic)
    out = []
    for r in d["rules"]:
        if not r.get("active", True):
            continue
        sc = r.get("scope") or "global"
        if sc == "global" or (t and _norm(sc) and _norm(sc)[:8] in t):
            out.append(r)
    return out


def preference_prompt(topic: str = "") -> str:
    """给文献审评 / 写作 Agent 看的人的偏好，人说了就算，不需要攒够次数。"""
    d = load_prefs()
    lines = []
    for r in active_rules(topic):
        verb = {"avoid": "不要选", "require": "优先选", "boost": "适当加分给"}[r["kind"]]
        lines.append(f"- {verb} {r['field']}={r['value']}"
                     + (f"（{r['note']}）" if r.get("note") else ""))
    drops = [x for x in d["paper_decisions"] if x.get("action") == "drop"]
    keeps = [x for x in d["paper_decisions"] if x.get("action") == "keep"]
    if drops:
        lines.append("- 明确排除过：" + "、".join(str(x.get("raw"))[:60] for x in drops[-5:]))
    if keeps:
        lines.append("- 明确要求保留：" + "、".join(str(x.get("raw"))[:60] for x in keeps[-5:]))
    if not lines:
        return ""
    return "【用户偏好（人工设定，优先级高于模型判断）】\n" + "\n".join(lines)


def review_notes_for(topic: str = "") -> list[dict]:
    """上一轮人对诊断报告的批注，注入本轮写作提示。"""
    d = load_prefs()
    out = []
    for x in d.get("review_notes", []):
        if x.get("status") == "rejected":
            out.append(f"评审意见「{str(x.get('detail'))[:80]}」已被人工驳回，不要再按它改")
        elif x.get("status") in ("accepted", "done"):
            out.append(f"人工确认要改：{str(x.get('detail'))[:120]}"
                       + (f"（{x.get('note')}）" if x.get("note") else ""))
        if len(out) >= 8:
            break
    return [{"detail": s} for s in out]


# ── 规则匹配 ──────────────────────────────────────────────────────────
def _match(rule: dict, p: dict) -> bool:
    field, val = rule.get("field"), str(rule.get("value") or "")
    if field == "id":
        hay = f"{p.get('id') or ''} {p.get('url') or ''} {p.get('title') or ''}"
        return _norm(val) in _norm(hay)
    if field == "title":
        return _norm(val) in _norm(p.get("title"))
    if field == "venue":
        return _norm(val) in _norm(p.get("venue"))
    if field == "topic":
        return _norm(val) in _norm(p.get("title")) or _norm(val) in _norm(p.get("abstract"))
    if field == "keyword":
        hay = f"{p.get('title') or ''} {p.get('abstract') or ''} {p.get('venue') or ''}".lower()
        return val.strip().lower() in hay
    if field == "year":
        y = str(p.get("year") or "")
        if not y.isdigit():
            return False
        m = re.match(r"(>=|<=|>|<|=)?\s*(\d{4})", val)
        if not m:
            return False
        op, n = m.group(1) or "=", int(m.group(2))
        yi = int(y)
        return {">=": yi >= n, "<=": yi <= n, ">": yi > n,
                "<": yi < n, "=": yi == n}[op]
    return False


def apply_preferences(papers: list[dict], topic: str = "") -> tuple[list[dict], dict]:
    """在入选集上应用人工偏好。只剔除与加分，不重排顺序（顺序 = 引用编号）。"""
    rules = active_rules(topic)
    decisions = load_prefs().get("paper_decisions", [])
    drop_keys = {x["key"] for x in decisions if x.get("action") == "drop"}
    keep_keys = {x["key"] for x in decisions if x.get("action") == "keep"}
    kept, removed, boosted = [], [], []
    for p in papers or []:
        if _paper_hits(drop_keys, p) and not _paper_hits(keep_keys, p):
            removed.append({"title": p.get("title"), "by": "human", "reason": "人工排除"})
            continue
        hit_avoid = [r for r in rules if r["kind"] == "avoid" and _match(r, p)]
        if hit_avoid and not _paper_hits(keep_keys, p):
            removed.append({"title": p.get("title"), "by": "human",
                            "reason": f"偏好规则 {hit_avoid[0]['field']}={hit_avoid[0]['value']}"})
            continue
        bonus = sum(float(r.get("weight") or 0) for r in rules
                    if r["kind"] in ("require", "boost") and _match(r, p))
        bonus += 0.3 if _paper_hits(keep_keys, p) else 0.0
        if bonus:
            p = dict(p)
            p["pref_bonus"] = round(bonus, 3)
            boosted.append({"title": p.get("title"), "bonus": round(bonus, 3)})
        kept.append(p)
    return kept, {"removed": removed, "boosted": boosted,
                  "rules_applied": len(rules), "n_before": len(papers or []),
                  "n_after": len(kept)}


# ── 本次选材台账 ──────────────────────────────────────────────────────
def attach_review(run_dir, review: dict) -> bool:
    """文献审评 Agent 跑完后，把「哪些该丢、缺什么」回填进本次台账。"""
    p = Path(run_dir) / LEDGER_NAME
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        d["review"] = {
            "coverage_score": review.get("coverage_score"),
            "assessment": review.get("assessment"),
            "weak_ids": review.get("weak_ids") or [],
            "redundant_ids": review.get("redundant_ids") or [],
            "missing_themes": review.get("missing_themes") or [],
            "recommended_add_queries": review.get("recommended_add_queries") or [],
            "provenance": review.get("provenance"),
        }
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def record_selection(topic: str, pool: list[dict], chosen: list[dict],
                     run_dir, selector: str = "model", applied: dict | None = None,
                     review: dict | None = None) -> dict:
    """把「候选 → 入选」的整条决策链写进 <out>/LIT_SELECTION.json。"""
    out = Path(run_dir)

    def _key(p: dict) -> str:
        # 候选池与入选集可能来自不同轮次（补抓会覆盖 01），两边 id 未必同形。
        # 统一只用 id（缺失时退回题名），避免同一篇被当成两条记进去。
        return _norm(p.get("id") or p.get("title"))

    chosen_keys = {_key(p) for p in (chosen or [])}
    decisions = load_prefs().get("paper_decisions", [])
    human_keys = {x["key"] for x in decisions}

    def _row(p: dict, selected: bool) -> dict:
        by = "human" if _paper_hits(human_keys, p) else (
            selector if selected else "rule")
        reason = (p.get("llm_why") or p.get("llm_what") or p.get("gist_cn") or
                  (f"role={p.get('role')}, fit={p.get('fit_score')}" if selected else
                   "未入选（贴合度/角色配额）"))
        return {
            "id": p.get("id") or _norm(p.get("title")),
            "title": (p.get("title") or "")[:160],
            "year": p.get("year"), "venue": p.get("venue"),
            "selected": selected, "by": by,
            "reason": str(reason)[:160],
        }

    rows, seen = [], set()
    for p in pool or []:
        k = _key(p)
        rows.append(_row(p, k in chosen_keys))
        seen.add(k)
    # 补抓/扩面后 01_literature.json 会被覆盖，而 02 候选池还是旧快照——
    # 此时入选文献可能不在候选里。不补上就会出现「入选 59 篇、台账只记 32 篇」。
    extra = 0
    for p in chosen or []:
        k = _key(p)
        if k not in seen:
            rows.append(_row(p, True))
            seen.add(k)
            extra += 1
    data = {
        "topic": topic, "run_dir": str(out), "ts": _now(),
        "n_candidates": len(pool or []), "n_selected": len(chosen or []),
        "n_extra_from_refetch": extra,
        "selector": selector,
        "preferences_applied": applied or {},
        "review": {"weak_ids": (review or {}).get("weak_ids") or [],
                   "missing_themes": (review or {}).get("missing_themes") or [],
                   "recommended_add_queries": (review or {}).get("recommended_add_queries") or []},
        "decisions": rows,
        "note": "by=human 表示人工对这篇表过态；selected 为 False 者是候选池中被剔除的条目。",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / LEDGER_NAME).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    return {"path": str(out / LEDGER_NAME), "n_candidates": len(pool or []),
            "n_selected": len(chosen or []), "n_human": sum(1 for r in rows if r["by"] == "human")}
