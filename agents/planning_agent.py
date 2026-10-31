#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PlanningAgent 薄壳（子代理拆分之一）。

封装 planner.plan_spec / iterate_spec，把生成或迭代出的 spec 写入 MemoryStore 的
project 命名空间（"spec" / "spec_round_N"），供 ExperimentAgent、WritingAgent 与
组员的 MemoryAgent（project memory）读取。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.store import MemoryStore, store as default_store  # noqa: E402
import planner as planner_mod  # noqa: E402


def run(topic: str, client=None, model=None, extra=None,
        data_override: dict | None = None, store: MemoryStore | None = None,
        out_path=None):
    """首轮规划：自然语言主题 -> spec，并写入 MemoryStore。"""
    store = store or default_store
    spec = planner_mod.plan_spec(topic, client, model, extra, data_override)
    store.put("project", "spec", {"topic": topic, "spec": spec})
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return spec


def iterate(prev_spec: dict, prev_result: dict, client=None, model=None,
            extra=None, round_idx: int = 2, store: MemoryStore | None = None):
    """迭代规划：把上一轮结果喂回 Planner，产出下一轮 spec。"""
    store = store or default_store
    nxt = planner_mod.iterate_spec(prev_spec, prev_result, client, model, extra, round_idx)
    if isinstance(nxt, dict):
        store.put("project", f"spec_round_{round_idx}", {"spec": nxt})
    return nxt
