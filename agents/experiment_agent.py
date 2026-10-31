#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExperimentAgent 薄壳（子代理拆分之一）。

封装 engine.analysis_engine 的真实实验运行，把 analysis 写入 MemoryStore 的
project 命名空间（"analysis"），供 WritingAgent 与组员读取。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.store import MemoryStore, store as default_store  # noqa: E402


def _load_engine():
    mod_spec = importlib.util.spec_from_file_location(
        "analysis_engine", ROOT / "engine" / "analysis_engine.py")
    ae = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(ae)
    return ae


def run(spec: dict, exp_path, store: MemoryStore | None = None):
    """运行确定性实验，返回 analysis 并写入 MemoryStore。"""
    store = store or default_store
    ae = _load_engine()
    exp_path = Path(exp_path)
    ds_list = spec.get("datasets") or spec.get("data_list") or []
    if ds_list and len(ds_list) >= 2:
        # 多数据集对比：run_multi_analysis 内部写 per-dataset 结果 + combined
        analysis = ae.run_multi_analysis(spec, exp_path)
    else:
        analysis = ae.run_analysis(spec, exp_path)
        # run_analysis 返回的就是扁平 ANALYSIS_JSON，直接落盘
        exp_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    store.put("project", "analysis", {
        "data": spec.get("data"),
        "task_type": spec.get("task_type"),
        "analysis": analysis,
    })
    return analysis
