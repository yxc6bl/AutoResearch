#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行追踪（Trace）——记录每一步「谁在什么时候做了什么、花了多久、成没成」。

为什么需要它：nanoMuse 那类系统论文里最有说服力的不是口号，而是**可审计的执行链**。
我们目前只有产物文件，没有「过程记录」：出问题时只能靠翻日志，说不清是哪个阶段、
哪次调用、耗时多少。这既不利于排障，也不利于写系统设计那一章。

产物：<out_dir>/00_trace.jsonl，一行一个阶段：
  {"ts","stage","status","duration_s","detail"}

用法：
  from engine import trace
  t0 = time.time(); ...; trace.record(out_dir, "analysis", time.time()-t0, "ok", "6 个方法")
  trace.summary(out_dir)  # 供 CLI 打印
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

FILE = "00_trace.jsonl"


def record(out_dir, stage: str, duration_s: float = 0.0,
           status: str = "ok", detail: str = "") -> dict:
    """追加一条阶段记录（只增不改，便于事后复算）。"""
    rec = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": stage,
        "status": status,
        "duration_s": round(float(duration_s or 0), 2),
        "detail": (detail or "")[:200],
    }
    try:
        p = Path(out_dir) / FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return rec


def load(out_dir) -> list[dict]:
    p = Path(out_dir) / FILE
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def summary(out_dir) -> dict:
    rows = load(out_dir)
    total = sum(r.get("duration_s") or 0 for r in rows)
    slowest = max(rows, key=lambda r: r.get("duration_s") or 0) if rows else None
    return {
        "n_stages": len(rows),
        "total_s": round(total, 1),
        "slowest": ({"stage": slowest["stage"], "duration_s": slowest["duration_s"]}
                    if slowest else None),
        "failed": [r["stage"] for r in rows if r.get("status") != "ok"],
        "stages": rows,
    }
