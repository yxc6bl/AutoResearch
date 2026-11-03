#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文献使用反馈：记录「哪些文献真的被用进论文」，并在后续检索里优先复用。

为什么需要它：文献筛选目前是一次性的——抓完、选完、写完就结束，系统不知道
哪篇文献真的支撑了正文、哪篇只是占了参考文献列表。没有这条反馈，筛选策略
永远无法根据实际效果调整。

记录内容（memory/lit_usage.json）：
  { <归一化标题>: {"uses": 被正文引用的次数, "runs": 入选次数,
                   "last_used": 时间, "roles": [角色...]} }

使用方式：
  - 检索阶段：对历史上「入选且被引用」的文献给正加分（上限 +0.10）；
  - 成稿之后：record() 统计本次每篇文献在正文被引用的次数并落盘。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USAGE_PATH = ROOT / "memory" / "lit_usage.json"


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", (t or "").lower())[:100]


def load() -> dict:
    try:
        return json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict) -> None:
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USAGE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def bonus_map() -> dict[str, float]:
    """历史被引用过的文献 → 排序加分（每被引用一次 +0.03，上限 0.10）。"""
    out = {}
    for key, rec in load().items():
        uses = int(rec.get("uses") or 0)
        if uses > 0:
            out[key] = min(0.10, 0.03 * uses)
    return out


def record(topic: str, papers: list[dict], paper_md: str) -> dict:
    """统计本次入选文献在正文中的被引次数，累加写入 history。返回本次明细。"""
    cited = set(int(x) for x in re.findall(r"\[(\d+)\]", paper_md or ""))
    data = load()
    details = []
    for i, p in enumerate(papers, start=1):
        title = p.get("title") or ""
        key = _norm_title(title)
        if not key:
            continue
        rec = data.setdefault(key, {"uses": 0, "runs": 0, "roles": [], "titles": []})
        rec["runs"] = int(rec.get("runs") or 0) + 1
        used = i in cited
        if used:
            rec["uses"] = int(rec.get("uses") or 0) + 1
        role = p.get("role")
        if role:
            roles = rec.setdefault("roles", [])
            if role not in roles:
                roles.append(role)
        rec["last_used"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        rec["last_topic"] = topic
        if title and title not in rec.setdefault("titles", []):
            rec["titles"] = (rec["titles"] + [title])[-3:]
        details.append({"num": i, "title": title, "role": role, "cited": used,
                        "fit": p.get("fit_score"), "score": p.get("select_score")})
    _save(data)
    return {
        "topic": topic,
        "n_selected": len(papers),
        "n_cited": sum(1 for d in details if d["cited"]),
        "details": details,
    }
