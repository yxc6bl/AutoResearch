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
register(BackendSpec("association", "transactions", ("association",),
                     deps=(),
                     note="独立分支 _run_association()：Apriori 频繁项集 + 规则生成"
                          "（支持度/置信度/提升度），纯 Python 无新依赖；source=association 交易 CSV"))
register(BackendSpec("graph", "graph", ("graph",),
                     deps=("numpy", "pandas", "sklearn"),
                     note="独立分支 _run_graph()：节点级分层 K 折 × 三种特征协议；"
                          "显式区分全图（含传导泄漏）与折内（无泄漏）邻居均值，"
                          "核心结论只从无泄漏协议中择优；source=graph 指 nodes+edges 两个文件"))
register(BackendSpec("image", "image", ("image",),
                     deps=("torch", "numpy", "matplotlib", "sklearn"),
                     note="独立模块 engine/image_backend.py：IDX 张量直读，"
                          "torch 小型 CNN / MLP + sklearn 线性基线，自管 batch/epoch/参数量；"
                          "跳过 build_tabular_preprocessor；source=image 指 IDX gzip 目录"))

# ── 尚未接线、按性价比排序的缺口 ────────────────────────────────────────
GAPS = [
    {
        "name": "领域数据缺失", "priority": 1, "new_branch": False,
        "topics": ["基于XGBoost的网络游戏流失玩家预测算法研究",
                   "基于机器学习的复合材料等效弹性性能研究"],
        "deps": ("pandas",), "extra": "本地 _concrete.data 是老式 .xls，读它要 xlrd 或 python-calamine；依赖状态看 dep_status，装没装以 --probe 为准",
        "how": "找公开 CSV → 放 data/ → DATASET_CATALOG 加一条 → 跑 benchmark。评估协议不变，不动 analysis_engine。",
    },
    {
        "name": "图/网络结构（已接线，代理数据）", "priority": 2, "new_branch": True,
        "topics": ["基于社交网络结构的社会推荐算法研究", "图神经网络过平滑问题的研究"],
        "deps": ("numpy",), "extra": "networkx 可选（装了更好读边表，不装纯 numpy 邻接矩阵也能做）；"
                              "2026-11-05 已接 Cora 真实引文网络，但这两道题仍是代理数据、且后端非真 GNN",
        "how": "已实现 source=graph（linqs/csv 两种格式）、_load_graph() 与 _run_graph()；"
               "剩余缺口是社交网络/推荐领域的同领域真实数据，以及真 GNN 的层数消融。",
    },
    {
        "name": "视觉/视频（小规模图像分类已接线）", "priority": 3, "new_branch": True,
        "topics": ["基于深度学习的车辆识别系统", "基于深度学习的智能无人收银系统",
                   "基于深度学习的交通标志检测系统", "基于深度神经网络的智能视频计算",
                   "基于迁移学习的模态检测方法", "基于机器学习的入侵检测方法"],
        "deps": ("torch", "torchvision"), "extra": "torch 2.15.0+cpu / torchvision 0.29.2+cpu 已装；"
                              "2026-11-08 已接 MNIST 小规模图像分类，但检测/分割/视频与真实场景数据仍是缺口",
        "how": "source=image 已实现（IDX 张量直读 + torch CNN/MLP + sklearn 线性基线）；"
               "剩余工作是把 DataLoader、增广、预训练权重与检测/视频指标接到同一套契约上。",
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


def probe(deps, timeout: float = 180.0) -> dict:
    """真导入一遍依赖，返回可用性、版本与耗时。

    `find_spec` 只能证明「文件在」，证明不了「import 得动」——torch 这类包
    会因为 DLL、ABI、Python 版本不匹配而装上却用不了。这里用子进程隔离，
    导入崩溃不会带崩 CLI，也能给出真实的冷启动耗时（写进论文的「环境」一节）。
    """
    import subprocess
    import sys
    import time

    out = {}
    for d in deps:
        t0 = time.time()
        code = (f"import json, {d} as m; "
                f"print(json.dumps({{'v': str(getattr(m, '__version__', 'unknown'))}}))")
        try:
            p = subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True, timeout=timeout)
            dt = round(time.time() - t0, 2)
            if p.returncode == 0:
                tail = (p.stdout or "").strip().splitlines()
                ver = ""
                if tail:
                    try:
                        ver = json.loads(tail[-1]).get("v", "")
                    except Exception:
                        ver = tail[-1][:40]
                out[d] = {"ok": True, "s": dt, "version": ver}
            else:
                err = ((p.stderr or "").strip().splitlines() or ["import 失败"])[-1]
                out[d] = {"ok": False, "s": dt, "error": err[:160]}
        except subprocess.TimeoutExpired:
            out[d] = {"ok": False, "s": round(time.time() - t0, 1),
                      "error": f"导入超时（>{timeout:.0f}s）"}
        except Exception as e:  # noqa: BLE001
            out[d] = {"ok": False, "s": round(time.time() - t0, 1), "error": str(e)[:160]}
    return out



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
