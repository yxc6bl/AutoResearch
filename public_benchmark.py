#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开论文 × 公开数据「同题基准」——把用户说的「只抓选题和数据，让我们自己跑」做成可复跑的链路。

思路：
  1) 从 Crossref 抓真实已发表论文的题录（题目 / DOI / 期刊 / 年份 / 摘要 / 被引数）→ 这是「原论文参考」；
  2) 与本地公开数据集目录（UCI / sklearn / OpenML）按主题配对 → 这是「公开实验数据」；
  3) 生成 spec 并跑确定性实验层，得到真实数字；
  4) 有生成稿时，与原文题录 + corpus/ 里的真实优秀范文做对比。

必须说清的边界（不许包装成「已经完成全文对标」）：
  Crossref 只给**元数据与摘要**，多数期刊全文有版权墙，拿不到全文。
  所以这里的参考层级是「元数据/摘要」，不是逐节全文对比。
  要做到全文级同题对比，得另抓 arXiv / PMLR / OpenReview 的开放获取 PDF，见 docs/GATES.md 未验证清单。

用法：
  python public_benchmark.py build                 # 抓真实选题 + 生成配对
  python public_benchmark.py list                  # 看配对清单
  python public_benchmark.py run --id 1           # 跑实验层（本地，不打模型网关）
  python public_benchmark.py compare --id 1       # 与原文/范文对比
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import topic_parser as tp  # noqa: E402
from agents import literature_agent as LA  # noqa: E402

PAIRS_FILE = ROOT / "corpus" / "pairs.json"
IMPORTS_DIR = ROOT / "corpus" / "imports"
OUT_ROOT = ROOT / "out_pairs"

# 公开数据集都来自公开来源，OpenML 那几个是运行时从 openml.org 真实拉取的 CSV。
# must / avoid 是配对语义门：Crossref 按关键词检索会把「虹膜识别」当成「鸢尾花」、
# 把「微波谐振器鉴别咖啡与酒」当成「葡萄酒品种分类」。宁可少配几对，也不制造假同题。
QUERIES = [
    {"query": "breast cancer classification random forest", "key": "breast_cancer",
     "hint": "乳腺癌良恶性分类，分类任务",
     "must": ["breast", "cancer", "tumor", "wdbc", "diagnosis"],
     "avoid": ["histopath", "mammograph", "microscopic", "ultrasound",
               "image", "imaging", "radiomic", "patholog"]},
    {"query": "heart disease prediction machine learning", "key": "heart",
     "hint": "心脏病预测，分类任务",
     "must": ["heart", "coronary", "cardiac"],
     "avoid": ["image", "echocardi", "segmentation"]},
    {"query": "iris flower classification support vector machine", "key": "iris",
     "hint": "鸢尾花品种分类，分类任务",
     "must": ["iris", "flower", "petal", "sepal"],
     "avoid": ["microscopic", "microscopy", "recognition", "eye", "biometric"]},
    {"query": "wine cultivar classification machine learning", "key": "wine",
     "hint": "葡萄酒品种鉴别，分类任务",
     "must": ["wine", "cultivar"],
     "avoid": ["resonator", "microwave", "coffee", "sensor", "spectroscopy"]},
    {"query": "mushroom classification decision tree", "key": "openml_mushroom",
     "hint": "蘑菇可食性判别，分类任务",
     "must": ["mushroom"], "avoid": ["image", "hyperspectral"]},
    {"query": "adult income prediction classification", "key": "openml_adult",
     "hint": "成人收入预测，分类任务",
     "must": ["income", "adult", "census"], "avoid": []},
    {"query": "bank marketing campaign classification", "key": "openml_bank",
     "hint": "银行营销响应预测，分类任务",
     "must": ["bank", "marketing"], "avoid": []},
    {"query": "wine quality prediction regression", "key": "openml_wine_quality",
     "hint": "葡萄酒质量评分，回归任务",
     "must": ["wine", "quality"],
     "avoid": ["resonator", "microwave", "coffee"]},
    {"query": "airfoil self noise prediction regression", "key": "airfoil",
     "hint": "翼型气动噪声预测，回归任务",
     "must": ["airfoil", "noise"], "avoid": []},
    {"query": "text classification newsgroups benchmark", "key": "text_sample",
     "hint": "中文文本分类，分类任务",
     "must": ["text", "classification"], "avoid": ["image", "video"]},
]


def _topic_rank(p: dict, must: list[str], avoid: list[str]) -> tuple[int, float]:
    """返回 (领域词命中数, 综合分)。命中排除词直接判 0 命中。"""
    blob = ((p.get("title") or "") + " " + (p.get("abstract") or "")).lower()
    if any(w in blob for w in avoid):
        return 0, 0.0
    hits = sum(1 for w in must if w in blob)
    if not hits:
        return 0, 0.0
    cited = p.get("cited_by_count") or 0
    import math
    # 领域命中数占主导，被引只作小幅加权：否则「高被引的图像论文」会压过
    # 「低被引但真正做表格分类的论文」，又制造一批假同题。
    score = hits * 2 + min(1.5, math.log10(1 + cited)) + (
        1.5 if len(p.get("abstract") or "") > 80 else 0.0)
    return hits, score

_DS = {d["key"]: d for d in tp.DATASET_CATALOG}


def _load_pairs() -> list[dict]:
    if not PAIRS_FILE.exists():
        return []
    try:
        return json.loads(PAIRS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_pairs(rows: list[dict]) -> None:
    PAIRS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAIRS_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def build(per_query: int = 5, per_topic: int = 2) -> int:
    """抓真实论文题录 + 配对公开数据集，写 corpus/pairs.json 与本地题录导入目录。"""
    rows: list[dict] = []
    all_papers: list[dict] = []
    for qi, item in enumerate(QUERIES, 1):
        query, key, hint = item["query"], item["key"], item["hint"]
        ds = _DS.get(key)
        if not ds:
            print(f"[build] 跳过：数据集 {key} 不在目录里")
            continue
        t0 = time.time()
        try:
            got = LA.fetch_crossref(query, max_results=per_query)
        except Exception as e:  # noqa: BLE001
            print(f"[build] {qi:2d}/{len(QUERIES)} Crossref 失败（{query[:34]}）：{str(e)[:80]}")
            continue
        n_before = len(got)
        ranked = []
        for p in got:
            hits, score = _topic_rank(p, item["must"], item["avoid"])
            if hits >= 2 or (hits >= 1 and len(p.get("abstract") or "") > 80):
                ranked.append((score, p))
        ranked.sort(key=lambda x: -x[0])
        got = [p for _, p in ranked]
        filtered = n_before - len(got)
        picked = 0
        for p in got:
            if not p.get("title") or picked >= per_topic:
                break
            try:
                spec = tp.assemble_title_spec(
                    p["title"], direction=hint,
                    data_override=tp._data_binding(ds))
            except Exception as e:  # noqa: BLE001
                print(f"[build] spec 生成失败：{e}")
                continue
            # 英文题目容易把回归题判成分类：以数据集本身的 task 为准，并如实记录。
            task_note = ""
            if spec.get("task_type") != ds["task"]:
                task_note = f"题目解析为 {spec.get('task_type')}，已按公开数据集的 {ds['task']} 校正"
                spec["task_type"] = ds["task"]
                spec["metric"] = "r2" if ds["task"] == "regression" else "accuracy"
            rows.append({
                "id": f"p{len(rows) + 1:02d}",
                "query": query,
                "reference": {
                    "title": p["title"], "doi": p.get("url", ""),
                    "venue": p.get("venue", ""), "year": p.get("year", ""),
                    "authors": p.get("authors", ""),
                    "abstract": p.get("abstract", ""),
                    "cited_by_count": p.get("cited_by_count", 0),
                    "level": "元数据/摘要（Crossref，无全文）",
                },
                "dataset": {"key": key, "label": ds["label"], "source": ds["source"],
                            "task": ds["task"], "public": True},
                "spec": spec, "task_note": task_note,
                "result": None, "comparison": None,
            })
            all_papers.append(p)
            picked += 1
        print(f"[build] {qi:2d}/{len(QUERIES)} {query[:38]:38s} → {picked} 对"
              f"（语义门剔除 {filtered} 条）｜{time.time() - t0:.1f}s")

    _save_pairs(rows)
    # 抓到的真实题录同时落进 local 源，下次检索直接复用，不依赖任何在线 API。
    if all_papers:
        IMPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (IMPORTS_DIR / "crossref_public_papers.json").write_text(
            json.dumps(all_papers, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[build] 共 {len(rows)} 对，写 {PAIRS_FILE}")
    print(f"[build] 真实题录 {len(all_papers)} 条 → {IMPORTS_DIR / 'crossref_public_papers.json'}")
    print("[build] 参考层级：元数据/摘要。全文级同题对比仍需开放获取 PDF，此处不冒充。")
    return len(rows)


def cmd_list(_args) -> int:
    rows = _load_pairs()
    if not rows:
        print("还没有配对，先跑 build")
        return 1
    for r in rows:
        mark = "已跑" if r.get("result") else "待跑"
        print(f"{r['id']} [{mark}] {r['reference']['title'][:68]}")
        print(f"      数据：{r['dataset']['label']}（{r['dataset']['task']}）｜"
              f"原文：{r['reference']['venue']} {r['reference']['year']}"
              f"｜被引 {r['reference']['cited_by_count']}")
    return 0


def cmd_run(args) -> int:
    rows = _load_pairs()
    hit = next((r for r in rows if r["id"] == args.id), None)
    if hit is None:
        print(f"没有 {args.id}，先跑 list")
        return 1
    out = OUT_ROOT / hit["id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "task_spec.json").write_text(
        json.dumps(hit["spec"], ensure_ascii=False, indent=2), encoding="utf-8")
    from agents import experiment_agent as EA
    t0 = time.time()
    try:
        res = EA.run(hit["spec"], out / "03_experiment.json")
    except Exception as e:  # noqa: BLE001
        print(f"[run] 实验失败：{e}")
        return 1
    if res.get("error"):
        print(f"[run] 实验失败：{res['error']}")
        return 1
    hit["result"] = {
        "best_method": res.get("best_method"), "best_value": res.get("best_value"),
        "n_metrics": len(res.get("metrics") or []),
        "seconds": round(time.time() - t0, 1),
        "out_dir": str(out),
    }
    for k in ("n_samples", "n_features"):
        if res.get(k) is not None:  # 该字段不是所有数据源都会返回，没有就不写，不伪造
            hit["result"][k] = res[k]
    _save_pairs(rows)
    print(f"[run] {hit['id']}：{res.get('best_method')} = {res.get('best_value')}"
          f"｜指标 {hit['result']['n_metrics']} 项｜耗时 {hit['result']['seconds']}s")
    print(f"      产物 {out / '03_experiment.json'}")
    print("      下一步：step --name write 成文（需模型网关），再 compare")
    return 0


def _numbers_in(text: str) -> list[float]:
    out = []
    for m in re.findall(r"(?<!\d)(0?\.\d{2,4}|\d{2}\.\d{1,2})\s*%?", text or ""):
        try:
            v = float(m)
            if 0 < v <= 100:
                out.append(v / 100 if v > 1.5 else v)
        except ValueError:
            continue
    return out[:8]


def cmd_compare(args) -> int:
    rows = _load_pairs()
    hit = next((r for r in rows if r["id"] == args.id), None)
    if hit is None:
        print(f"没有 {args.id}")
        return 1
    out = OUT_ROOT / hit["id"]
    md = out / "06_paper" / "paper_full.md"
    rep = {
        "pair_id": hit["id"],
        "reference_level": hit["reference"]["level"],
        "reference": {k: hit["reference"][k] for k in
                      ("title", "venue", "year", "doi", "cited_by_count")},
        "dataset": hit["dataset"],
        "experiment": hit.get("result"),
        "title_source": "生成稿题目沿用原论文题目录入（同题，非改写）",
        "has_generated_paper": md.exists(),
        "limitations": [
            "Crossref 只有元数据与摘要，不是全文；不能做逐节、逐公式的内容对标。",
            "原文与本次运行的划分方式、预处理、超参数不同，数值只能看量级，不能直接比高低。",
        ],
    }
    if hit.get("result") and hit["reference"].get("abstract"):
        nums = _numbers_in(hit["reference"]["abstract"])
        rep["reference_reported_numbers"] = nums
        rep["note_on_numbers"] = ("摘要中出现的数字（若有）仅作量级参照；"
                                  "不同实验协议的数值不具备可比性，不得据此判定优劣。")
    if md.exists():
        try:
            import benchmark_corpus as BC
            rep["corpus_benchmark"] = BC.benchmark(str(md), str(ROOT / "corpus"))
        except Exception as e:  # noqa: BLE001
            rep["corpus_benchmark_error"] = str(e)
    else:
        rep["next"] = (f"python auto_research.py step --name write --out {out} && "
                       f"python public_benchmark.py compare --id {hit['id']}")
    hit["comparison"] = rep
    _save_pairs(rows)
    (out / "comparison.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in
                      ("reference_level", "reference", "title_source", "experiment",
                       "has_generated_paper", "reference_reported_numbers", "next")
                      if k in rep},
                     ensure_ascii=False, indent=2))
    print(f"\n完整报告：{out / 'comparison.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="公开论文 × 公开数据 同题基准")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build"); b.add_argument("--per-query", type=int, default=5)
    b.add_argument("--per-topic", type=int, default=2)
    sub.add_parser("list")
    r = sub.add_parser("run"); r.add_argument("--id", required=True)
    c = sub.add_parser("compare"); c.add_argument("--id", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        n = build(a.per_query, a.per_topic)
        return 0 if n else 1
    if a.cmd == "list":
        return cmd_list(a)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "compare":
        return cmd_compare(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
