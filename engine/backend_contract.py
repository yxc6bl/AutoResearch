#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""领域后端契约（Backend Contract）——回答「补一个后端到底要改哪几处」。

这个系统里的「后端」不是一个独立模块，而是散在四个地方的接线点：

  ① 题目解析   engine/topic_parser.py  DATASET_CATALOG / _data_binding / match_dataset
  ② 数据加载   engine/analysis_engine.py  _load_dataset_inner() → (X, y, feat, task)
  ③ 评估协议   engine/analysis_engine.py  run_analysis() 的分支 + build_estimator()
  ④ 事实契约   ANALYSIS_JSON 字段；eval_agent._ground_truth() 与
               generate_paper._fact_pack() 只认这些字段，字段齐了论文侧不用改。

所以补后端 = 实现「数据怎么变成 X/y」+「怎么 fit/eval」，并保证产物符合
ANALYSIS_JSON 契约；题目解析加一条目录；论文与评审侧原则上零改动。

本模块不改变主流程，只提供：注册表、依赖自检、ANALYSIS_JSON 契约校验、缺口清单。
配套操作手册见 docs/BACKENDS.md。
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

# 论文写作端 / LLM 评审官实际会读的字段。新后端必须给齐，否则正文会写「素材未提供」，
# 评审官也会把缺失当成编造。
REQUIRED_FIELDS = ("task_type", "best_method", "best_value", "metrics", "plots")
RECOMMENDED_FIELDS = ("significance", "seed_variance", "preprocessing",
                      "confusion_matrix", "per_class", "findings", "adaptations")

WIRING = {
    "topic": "engine/topic_parser.py: DATASET_CATALOG / _data_binding() / detect_task_type()",
    "load": "engine/analysis_engine.py: _load_dataset_inner() 按 spec.data.source 分派",
    "eval": "engine/analysis_engine.py: run_analysis() 分支 / build_estimator() 方法工厂",
    "fact": "eval_agent._ground_truth() 与 generate_paper._fact_pack() 消费 ANALYSIS_JSON",
}


@dataclass
class BackendSpec:
    name: str
    modality: str
    sources: tuple[str, ...]
    deps: tuple[str, ...] = ()
    status: str = "wired"          # wired / planned
    wiring: dict = field(default_factory=lambda: dict(WIRING))
    note: str = ""


_REGISTRY: dict[str, BackendSpec] = {}


def register(spec: BackendSpec) -> BackendSpec:
    _REGISTRY[spec.name] = spec
    return spec


# ── 已经接线、跑通过的后端 ──────────────────────────────────────────────
register(BackendSpec("tabular", "tabular", ("sklearn", "csv", "parquet", "excel",
                                            "feather", "url", "openml", "内置名"),
                     deps=("pandas", "sklearn"),
                     note="主流程：分类/回归，多种子 + 显著性 + 消融；build_tabular_preprocessor 假设 X 是 DataFrame"))
register(BackendSpec("text_tfidf", "text", ("text",),
                     deps=("pandas", "sklearn"),
                     note="独立分支 _run_text()：TF-IDF 在 CV 折内拟合，防泄漏；X 是原始文档列表，不是 DataFrame"))
register(BackendSpec("timeseries", "timeseries", ("timeseries",),
                     deps=("pandas", "numpy"),
                     note="滑窗 lag + 季节编码把序列转成表格，之后复用主流程回归分支，无需新评估协议"))
register(BackendSpec("clustering", "tabular", ("sklearn",),
                     deps=("sklearn",),
                     note="独立分支：不做 train/test 划分，全量拟合，ARI/NMI/silhouette 评估"))

# ── 尚未接线、按性价比排序的缺口 ────────────────────────────────────────
GAPS = [
    {
        "name": "领域数据缺失", "priority": 1, "new_branch": False,
        "topics": ["基于XGBoost的网络游戏流失玩家预测算法研究",
                   "基于机器学习的复合材料等效弹性性能研究"],
        "deps": ("pandas",), "extra": "本地 _concrete.data 是老式 .xls，需 xlrd 或 python-calamine（当前都没装）",
        "how": "找公开 CSV → 放 data/ → DATASET_CATALOG 加一条 → 跑 benchmark。评估协议不变，不动 analysis_engine。",
    },
    {
        "name": "图/网络结构", "priority": 2, "new_branch": True,
        "topics": ["基于社交网络结构的社会推荐算法研究", "图神经网络过平滑问题的研究"],
        "deps": ("numpy",), "extra": "networkx 可选（当前没装）；纯 numpy 邻接矩阵也能做",
        "how": "新增 source=graph，load 返回节点特征 + 邻接矩阵 + 标签；新增节点分类分支（不能套独立样本随机划分，要用节点级划分）；指标用 accuracy/Macro-F1。",
    },
    {
        "name": "视觉/视频", "priority": 3, "new_branch": True,
        "topics": ["基于深度学习的车辆识别系统", "基于深度学习的智能无人收银系统",
                   "基于深度学习的交通标志检测系统", "基于深度神经网络的智能视频计算",
                   "基于迁移学习的模态检测方法", "基于机器学习的入侵检测方法"],
        "deps": ("torch", "torchvision"), "extra": "两者当前都没装；还需预训练权重与 CPU/GPU 预算",
        "how": "新增 source=image，load 返回张量；run_analysis 增图像分支（跳过 build_tabular_preprocessor，它对张量不适用）；评估用 top-1/mAP；事实清单必须给真实 epoch/权重来源。",
    },
    {
        "name": "文本/语言模型", "priority": 4, "new_branch": True,
        "topics": ["基于知识图谱的农业知识问答系统", "基于检索增强的大语言模型算法研究"],
        "deps": ("transformers", "torch"), "extra": "权重几百 MB~数 GB；且这两道题本质是研究方法/系统题，不是跑分题",
        "how": "若只做语义分类，可在 _run_text 旁加句向量分支；若做 RAG/问答，需要检索库 + 生成评估，超出确定性跑分流水线的定位，不建议塞进本引擎。",
    },
]


def specs() -> list[BackendSpec]:
    return list(_REGISTRY.values())


def check_deps(deps) -> dict:
    return {d: bool(importlib.util.find_spec(d)) for d in deps}


def gaps() -> list[dict]:
    out = []
    for g in GAPS:
        out.append({**g, "dep_status": check_deps(g["deps"]),
                    "deps_ready": all(check_deps(g["deps"]).values())})
    return out


def validate_analysis(analysis: dict) -> tuple[bool, list[str]]:
    """校验一个后端的产物能不能被论文/评审侧正确消费。"""
    issues: list[str] = []
    if not isinstance(analysis, dict):
        return False, ["ANALYSIS_JSON 不是 dict"]
    if analysis.get("error"):
        return False, [f"产物带 error：{analysis['error']}"]
    for k in REQUIRED_FIELDS:
        if k not in analysis:
            issues.append(f"缺必需字段 {k}")
    metrics = analysis.get("metrics") or []
    if not metrics:
        issues.append("metrics 为空（正文会写不出真实数字）")
    for m in metrics[:20]:
        if not isinstance(m, dict) or not isinstance(m.get("value"), (int, float)):
            issues.append(f"metrics 条目缺数值 value：{str(m)[:60]}")
            break
    for k in RECOMMENDED_FIELDS:
        if k not in analysis:
            issues.append(f"建议补字段 {k}（缺省不致命，但正文/评审会少证据）")
    return (not any(i.startswith("缺必需") or "metrics" in i for i in issues)), issues


def validate_file(path) -> tuple[bool, list[str]]:
    p = Path(path)
    if not p.exists():
        return False, [f"找不到 {p}"]
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return False, [f"JSON 解析失败：{e}"]
    return validate_analysis(d)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="领域后端契约自检")
    ap.add_argument("--check", default="", help="校验某个 03_experiment.json")
    a = ap.parse_args()
    print("接线点：")
    for k, v in WIRING.items():
        print(f"  {k:5s} {v}")
    print("\n已接线后端：")
    for s in specs():
        print(f"  {s.name:12s} source={','.join(s.sources)} deps={dict(check_deps(s.deps))}")
    print("\n缺口（按性价比）：")
    for g in gaps():
        print(f"  P{g['priority']} {g['name']}｜依赖就绪={g['deps_ready']}"
              f"｜需新分支={g['new_branch']}")
        print(f"     {g['how']}")
    if a.check:
        ok, issues = validate_file(a.check)
        print(f"\n契约校验 {a.check}：{'通过' if ok else '未通过'}")
        for i in issues:
            print("  -", i)
