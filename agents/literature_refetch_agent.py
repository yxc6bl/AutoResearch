#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文献缺口补抓 Agent —— 把「审评意见」真正接回检索端，形成闭环。

此前三个文献环节是断的：
  GatherAgent 抓+读 → ReviewAgent 审覆盖/缺口 → 04_literature_review.json 落盘 → 结束。
审评给出的 recommended_add_queries / missing_themes 从来没有人再拿去搜一遍；
而且 GatherAgent 内部两轮补抓（缺口回填、覆盖不足扩面）新增的文献**没有经过
LM 逐篇读摘要**，所以实测会出现 28/45 篇带阅读判断、剩下 17 篇纯关键词兜底。

本模块做两件事：
  1) 把审评缺口转成检索式再抓一轮，去重后合并；
  2) 新增文献同样走 _lm_read_screen（复用 0x_screen_cache.json，不重复花钱），
     补上 llm_what / llm_usable_in / confidence，重新按角色覆盖精选并写回 01_literature.json。

产物：0x_gap_refetch.json（补抓前后数量、新增、阅读覆盖、使用的检索式），
      01_literature.json（覆盖为重新精选后的文献集，供写作端使用）。

可单独调用：
  python agents/literature_refetch_agent.py --out DIR --topic "题目"
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import llm_client as LLM  # noqa: E402
from agents import literature_agent as LA  # noqa: E402
from agents import literature_gather_agent as G  # noqa: E402

REPORT_FILE = "0x_gap_refetch.json"


def _clean_query(q: str) -> str:
    """把 LM 给的布尔检索式洗干净再喂 Crossref / OpenAlex。

    LM 爱输出 `"A" AND ("B" OR "C") AND (2022:2025)` 这种语法，但这两个 API 的
    bibliographic / search 参数并不解析布尔表达式，原样送过去召回会非常差。
    """
    s = re.sub(r"\b(AND|OR|NOT|NEAR)\b", " ", q, flags=re.I)
    s = re.sub(r"[()\"'?:,]", " ", s)
    s = re.sub(r"\b\d{4}\b", " ", s)
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", s)
    stop = {"the", "of", "for", "in", "on", "with", "and", "or", "a", "an", "based"}
    words = [w for w in dict.fromkeys(words) if w.lower() not in stop]
    return " ".join(words[:8]) or q.strip()[:80]


def queries_from_review(review: dict, topic: str, max_queries: int = 4) -> list[str]:
    """审评意见 → 检索式。LM 给的先清洗；中文主题用规则抽词，不再额外打一次模型。"""
    out: list[str] = []
    for q in (review.get("recommended_add_queries") or []):
        if isinstance(q, str) and 3 < len(q.strip()) < 220:
            cq = _clean_query(q)
            if cq and cq not in out:
                out.append(cq)
    for theme in (review.get("missing_themes") or []):
        if not isinstance(theme, str):
            continue
        en = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", theme)
        cn = re.findall(r"[\u4e00-\u9fff]{2,10}", theme)
        if en:
            q = " ".join(dict.fromkeys(en))[:120]
        elif cn:
            # 中文主题：保留原短语并补英文通用词，Crossref 对纯中文召回有限，
            # 但检索式会原样写进报告，能看出这一轮到底搜了什么。
            q = " ".join(dict.fromkeys(cn[:4])) + " survey machine learning"
        else:
            continue
        q = q.strip()[:140]
        if q and q not in out:
            out.append(q)
    if not out and topic:
        base = topic.strip()[:60]
        out = [base + " survey", base + " benchmark"]
    return out[:max_queries]


def run(topic: str, out_dir, review: dict, papers: list[dict],
        budget_sec: float = 90, top_n: int = 60,
        max_queries: int = 4) -> dict:
    """按审评缺口补抓一轮，并把精选结果写回 01_literature.json。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    queries = queries_from_review(review or {}, topic, max_queries)
    rep = {
        "topic": topic, "queries": queries,
        "raw_queries": [q for q in ((review or {}).get("recommended_add_queries") or [])
                        if isinstance(q, str)][:max_queries],
        "n_before": len(papers or []), "n_added": 0, "n_read_added": 0,
        "n_after": len(papers or []), "read_coverage_before": 0.0,
        "read_coverage_after": 0.0, "seconds": 0.0,
        "status": "skipped", "reason": "",
    }
    if not queries:
        rep["reason"] = "审评未给出可用缺口检索式"
        _write(out_dir, rep)
        return rep

    papers = list(papers or [])
    known = {(p.get("id") or p.get("url")) for p in papers if p.get("id") or p.get("url")}
    deadline = time.time() + max(20.0, float(budget_sec))
    try:
        extra = G._fetch_pool(queries, per_query=25, deadline=deadline)
    except Exception as e:  # noqa: BLE001
        rep["reason"] = f"补抓失败：{e}"
        _write(out_dir, rep)
        return rep
    added_all = [p for p in extra if (p.get("id") or p.get("url")) not in known]
    # 只把质量分最高的一批送去重读。实测第一次补抓拿到 288 条，时间预算内
    # 只读了 13 篇，等于往池子里倒了一堆「没读过」的文献，反而稀释可信度。
    added = sorted(added_all, key=lambda p: -(p.get("quality_score") or 0))[:36]
    rep["n_candidates_fetched"] = len(extra)
    rep["n_added_raw"] = len(added_all)
    rep["n_added"] = len(added)

    # 关键：补抓来的也要逐篇读，不能只靠关键词兜底混进正文
    if added and LLM.available():
        try:
            screen = G._lm_read_screen(topic, added, out_dir, deadline=deadline,
                                       max_papers=36, batch=12, max_calls=3)
            rep["n_read_added"] = G._apply_screen(added, screen) if screen else 0
        except Exception as e:  # noqa: BLE001
            rep["read_error"] = str(e)[:120]
    for p in added:
        p.setdefault("confidence", "keyword_only")

    def _cov(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        n = sum(1 for p in rows if p.get("confidence") == "lm_read"
                or p.get("llm_what") or p.get("llm_fit") is not None)
        return round(n / len(rows), 3)

    rep["read_coverage_before"] = _cov(papers)
    pool = papers + added
    fit_topic = (topic or "").strip()
    try:
        chosen, sel = LA.select_by_role(pool, fit_topic, max_total=top_n)
        if not chosen:
            chosen, sel = LA.select_by_role(pool, fit_topic, max_total=top_n,
                                            profile="recall")
            sel["fallback"] = "补抓后全部低于质量线，已放宽一档"
    except Exception as e:  # noqa: BLE001
        rep["reason"] = f"重新精选失败：{e}"
        _write(out_dir, rep)
        return rep

    # 入选补读：补抓后篇数变多，若不给最终入选的文献补读，覆盖率反而会下降
    # （实测 45→60 篇时从 62.2% 掉到 61.7%）。只对入选且未读过的补，数量可控。
    _todo = [p for p in chosen if not (p.get("llm_what") or p.get("llm_fit") is not None)]
    if LLM.available() and _todo:
        try:
            _sc2 = G._lm_read_screen(topic, _todo[:24], out_dir,
                                     deadline=deadline + 60,
                                     max_papers=24, batch=12, max_calls=2)
            if _sc2:
                rep["n_read_selected"] = G._apply_screen(_todo[:24], _sc2)
        except Exception as e:  # noqa: BLE001
            rep["selected_read_error"] = str(e)[:120]

    (out_dir / "01_literature.json").write_text(
        json.dumps({"papers": chosen}, ensure_ascii=False, indent=2), encoding="utf-8")
    rep["n_after"] = len(chosen)
    rep["read_coverage_after"] = _cov(chosen)
    rep["selection"] = {k: sel.get(k) for k in
                        ("mode", "profile", "n_selected", "n_eligible",
                         "capped_by_cap", "dropped") if k in sel}
    rep["status"] = "ok" if added else "no_new_papers"
    rep["seconds"] = round(time.time() - t0, 1)
    _write(out_dir, rep)
    print(f"[RefetchAgent] 缺口补抓：{len(added)} 条新候选，逐篇阅读 "
          f"{rep['n_read_added']} 篇；精选 {len(chosen)} 篇"
          f"（LM 阅读覆盖 {rep['read_coverage_before']:.0%} → {rep['read_coverage_after']:.0%}）")
    return rep


def _write(out_dir: Path, rep: dict) -> None:
    try:
        (out_dir / REPORT_FILE).write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="按文献审评缺口补抓")
    ap.add_argument("--out", required=True, help="产物目录（需有 04_literature_review.json）")
    ap.add_argument("--topic", default="", help="研究选题")
    ap.add_argument("--budget", type=float, default=90)
    a = ap.parse_args()
    d = Path(a.out)
    rev = json.loads((d / "04_literature_review.json").read_text(encoding="utf-8")) \
        if (d / "04_literature_review.json").exists() else {}
    cur = json.loads((d / "01_literature.json").read_text(encoding="utf-8")) \
        if (d / "01_literature.json").exists() else {}
    rows = cur.get("papers") or [] if isinstance(cur, dict) else cur
    print(json.dumps(run(a.topic, d, rev, rows, budget_sec=a.budget),
                     ensure_ascii=False, indent=2))
