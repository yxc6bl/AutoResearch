"""Auto Research 统一入口（agent 喂 spec 的接口）。

设计目标：把系统从「手写三个演示任务」升级为「通用科研流水线」。
外部 agent / 人类研究者只需提供一份任务 spec（JSON），调用：

    python auto_research.py run --spec path/to/spec.json --out ./out_mytask

即可完成「确定性分析 → 出图 → 生成中文论文」，并把产物目录返回。
引擎对数据集与方法零硬编码：spec 里声明什么就跑什么。

spec 契约见 SPEC_SCHEMA.md。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS_ENGINE = ROOT / "engine" / "analysis_engine.py"
GEN_PAPER = ROOT / "paper" / "generate_paper.py"

# 忠实选题：题目语义决定是否注入「方法贡献」。网关/模块缺失时保守默认 True（给轻量贡献）。
try:
    from engine import topic_parser as _tp
except Exception:  # noqa: BLE001
    _tp = None


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env_with_dotenv() -> dict:
    """把项目 .env 注入环境变量（供论文生成器 / Planner 调用 LLM 网关）。"""
    env = dict(os.environ)
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            # 不覆盖已存在的环境变量：网页注入的用户 Key 优先于服务端 .env
            env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def _dataset_meta(data_override: dict | None) -> str | None:
    """为自定义数据集算一段「真实概况」，喂给 Planner 写出与领域一致的研究问题。

    不读全量数据，仅预览前若干行抽取列名与目标列分布，足够让 LLM 定位领域。
    """
    if not data_override:
        return None
    src = data_override.get("source")
    if src == "csv":
        p = data_override.get("path")
        if not p or not Path(p).exists():
            return None
        try:
            import pandas as pd
            df = pd.read_csv(p, nrows=200)
        except Exception:
            return None
    elif src == "url":
        return "（用户提供 URL 数据集，运行时下载；领域以研究主题为准）"
    else:
        return None
    target = data_override.get("target")
    tinfo = ""
    if target and target in df.columns:
        try:
            vc = df[target].value_counts()
            tinfo = f"；目标列「{target}」取值分布：{dict(vc)}"
        except Exception:
            tinfo = ""
    feats = [c for c in df.columns if c != target][:15]
    return (f"自定义表格数据集：预览 {len(df)} 行、共 {df.shape[1]} 列；"
            f"特征列含 {feats}{tinfo}。")


def run_task(spec: dict, out_dir, make_paper: bool = True, mode: str = "general",
             force: bool = False) -> dict:
    """程序化 API：给定 spec，端到端产出分析 + 论文。

    返回 dict：{analysis, out_dir, paper(路径或None), errors(list)}
    mode: general（默认，通用学术标准）或 sztu（深圳技术大学预设）。
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 写 spec 副本（论文生成器要读标题 / 指标 / 假设等）
    # 写入机构模式，供 generate_paper 决定封面/诚信声明措辞
    spec = dict(spec)
    spec["institution_mode"] = mode

    # 忠实选题：是否需要注入「本文方法贡献」（融合集成）。
    # 题目语义决定（detect_contribution_intent）：「基于随机森林」类忠实复现 → 不注入，
    # 论文忠于题目围绕随机森林做严谨实证，不再硬塞一个集成贡献而跑题；
    # 「改进/融合/提出」类 → 注入，作为本文贡献点。spec 已自带 proposed_method 则尊重之。
    _needs_contrib = spec.get("_needs_contribution", None)
    if _needs_contrib is None and _tp is not None:
        _needs_contrib = _tp.detect_contribution_intent(
            spec.get("title") or spec.get("topic") or "",
            spec.get("direction") or "")
    if _needs_contrib is None:
        _needs_contrib = True
    spec["_needs_contribution"] = _needs_contrib

    if "proposed_method" not in spec and _needs_contrib:
        _cms = spec.get("candidate_methods") or []
        _base_keys = []
        for _m in _cms:
            if isinstance(_m, str):
                _base_keys.append(_m)
            elif isinstance(_m, dict):
                _base_keys.append(_m.get("name") or _m.get("estimator")
                                  or f"m{len(_base_keys)}")
        if len(_base_keys) >= 2:
            spec["proposed_method"] = {
                "name": "本文提出的融合集成方法",
                "type": "voting",
                "voting": "soft",
                "base": _base_keys,
            }
    elif not _needs_contrib:
        print("[选题] 忠实复现模式：题目未要求方法创新，不注入融合集成贡献，"
              "论文围绕题目点名方法做严谨实证对比。")
    (out_dir / "task_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2) 文献：LM 驱动检索 Agent（真正理解选题去搜+筛）→ 文献审评 Agent（第二双眼睛）
    lit_path = out_dir / "01_literature.json"
    _need_fetch = False
    if not lit_path.exists():
        _need_fetch = True
    else:
        try:
            _cur = json.loads(lit_path.read_text(encoding="utf-8"))
            if not _cur.get("papers"):
                _need_fetch = True
        except Exception:
            _need_fetch = True
    # 断点续跑：已存在文献精选集且未强制 --force 时，跳过（重抓会重复打爆网关、易触发 429）
    if lit_path.exists() and not force:
        _need_fetch = False
        print(f"[literature] 复用已有 {lit_path.name}（--force 可强制重抓）")
    if _need_fetch:
        try:
            ga = _load_module(ROOT / "agents" / "literature_gather_agent.py", "lg_agent")
            _topic = (spec.get("title") or spec.get("topic") or spec.get("direction")
                      or spec.get("research_question") or spec.get("literature_query") or "")
            _methods = [m.get("name") for m in (spec.get("candidate_methods") or [])
                        if isinstance(m, dict)]
            _dq = spec.get("literature_query") or ""
            _gres = ga.run(_topic, task_type=spec.get("task_type", "classification"),
                           method_cn=_methods, dataset_query=_dq, out_dir=out_dir, top_n=45)
            print(f"[GatherAgent] {'LM驱动' if _gres['lm_driven'] else '规则兜底'}检索，"
                  f"精选 {len(_gres['papers'])} 篇文献")
            # 第二双眼睛：文献审评 Agent，独立判断覆盖度/冗余/缺口
            try:
                ra = _load_module(ROOT / "agents" / "literature_review_agent.py", "lr_agent")
                _cands = _gres.get("candidates") or []
                _lrev = ra.run(_topic, _gres["papers"], _cands)
                (out_dir / "04_literature_review.json").write_text(
                    json.dumps(_lrev, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[ReviewAgent] 文献覆盖评分 {_lrev.get('coverage_score')}/10"
                      f"（{'LM' if _lrev.get('lm_driven') else '规则'}）；"
                      f"缺口：{_lrev.get('missing_themes') or '无'}")
            except Exception as e:  # noqa: BLE001
                print(f"[ReviewAgent] 文献审评失败（不影响主流程）：{e}")
        except Exception as e:  # noqa: BLE001
            print(f"[literature] 检索失败（不影响主流程）：{e}")
            if not lit_path.exists():
                lit_path.write_text(json.dumps({"papers": []}, ensure_ascii=False), encoding="utf-8")
    hyps = {
        "sub_hypotheses": spec.get("sub_hypotheses", []),
        "hypothesis_statement": spec.get("hypothesis_statement", ""),
        "research_question": spec.get("research_question", ""),
    }
    (out_dir / "02_hypotheses.json").write_text(
        json.dumps(hyps, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3) 确定性分析（ExperimentAgent 封装，写 project memory）
    result: dict = {"analysis": None, "out_dir": str(out_dir), "paper": None, "errors": []}
    exp_path = out_dir / "03_experiment.json"
    try:
        ea = _load_module(ROOT / "agents" / "experiment_agent.py", "experiment_agent")
        analysis = ea.run(spec, exp_path)
    except Exception as e:  # noqa: BLE001
        result["errors"].append("实验分析失败: " + str(e))
        return result

    result: dict = {"analysis": analysis, "out_dir": str(out_dir), "paper": None, "errors": []}
    if analysis.get("error"):
        result["errors"].append("分析失败: " + str(analysis["error"]))
        return result

    # 4) 生成论文 + 评审闭环（WritingAgent 写作 → EvalAgent 的 LLM 评审官审 → 回滚重写）
    #    一个人写、另一个人审：LLM 评审官审不过（verdict=revise）就把意见写成 04_review.json，
    #    写作端下一轮带意见重写，最多 _MAX_ROUNDS 轮，取 LLM 评审分最高的一版。网关不可用时
    #    LLM 评审官不可用，自动退化为单轮规则评分（旧行为），不阻塞主流程。
    if make_paper:
        try:
            wa = _load_module(ROOT / "agents" / "writing_agent.py", "writing_agent")
            ea = _load_module(ROOT / "agents" / "eval_agent.py", "eval_agent")
            _md = out_dir / "06_paper" / "paper_full.md"
            _litc = out_dir / "01_literature.json"
            _cands = None
            if _litc.exists():
                try:
                    _d = json.loads(_litc.read_text(encoding="utf-8"))
                    _cands = _d if isinstance(_d, list) else (_d.get("papers") or [])
                except Exception:
                    _cands = None
            # 清掉上一轮可能残留的评审简报，保证首轮写作不带历史意见
            _review_brief = out_dir / "04_review.json"
            if _review_brief.exists():
                try:
                    os.remove(_review_brief)
                except Exception:
                    pass

            _MAX_ROUNDS = 2
            _verdict = None
            _best = None
            for _rnd in range(_MAX_ROUNDS):
                # 写作步对网关限流敏感：瞬时 429 导致子进程非零退出时，退避后重试一次
                _wres = None
                for _wattempt in range(2):
                    _wres = wa.run(out_dir, out_dir / "task_spec.json", timeout=1800)
                    if _wres["returncode"] == 0:
                        break
                    print(f"[WritingAgent] 第 {_wattempt + 1} 次写作返回非零（可能网关限流），"
                          f"退避 20s 后重试…")
                    time.sleep(20)
                if _wres["returncode"] != 0:
                    result["errors"].append("generate_paper 失败: " + (_wres["stderr"] or ""))
                    break
                paper = out_dir / "06_paper" / "paper_full.docx"
                result["paper"] = str(paper) if paper.exists() else None
                if not _md.exists():
                    break
                # LLM 评审官（含确定性硬门）打分
                _verdict = ea.run(str(_md), lit_candidates=_cands, mode=mode,
                                 out_path=out_dir / "06_paper" / "07_eval_verdict.json")
                result["eval"] = _verdict
                _lj = _verdict.get("llm_judge")
                _ok = "[通过]" if _verdict["passed"] else "[未通过(建议重做)]"
                _ljs = (f" | LLM评审 {_lj['overall']}" if (_lj and _lj.get("available")) else "")
                print(f"[EvalAgent] 质量评分门：{_ok} | 综合 {_verdict['overall']} | "
                      f"硬门 {_verdict['gate']} | 文献 {_verdict.get('literature_grade')}{_ljs}")
                # 记录最优（优先 LLM 评审分，无 LLM 时用规则综合分）
                _score = (_lj.get("overall") if (_lj and _lj.get("available")) else _verdict["overall"])
                if _best is None or _score > _best["score"]:
                    _best = {"score": _score, "round": _rnd + 1}
                # 评审官要求修订且还有轮次 → 写回滚简报，下一轮带意见重写
                if (_rnd < _MAX_ROUNDS - 1 and _lj and _lj.get("available")
                        and _lj.get("verdict") == "revise"):
                    _review_brief.write_text(json.dumps({
                        "unsupported_claims": _lj.get("unsupported_claims", []),
                        "suggestions": _lj.get("suggestions", []),
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"[EvalAgent] LLM评审官建议修订，回滚重写第 {_rnd + 2} 轮…")
                    continue
                break
            if _verdict is not None and not _verdict["passed"]:
                print(f"[EvalAgent] 主要问题：{'; '.join(_verdict['issues'][:5])}")
            if _best is not None:
                print(f"[EvalAgent] 评审闭环完成：最优为第 {_best['round']} 轮（评分 {_best['score']}）")
        except Exception as e:  # noqa: BLE001
            result["errors"].append(f"generate_paper 异常: {e}")

    return result


def cmd_run(args) -> int:
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    res = run_task(spec, args.out, make_paper=not args.no_paper, mode=args.mode)
    a = res["analysis"]
    summary = {
        "out_dir": res["out_dir"],
        "paper": res["paper"],
        "best_method": a.get("best_method"),
        "best_value": a.get("best_value"),
        "plots": [p.get("file") for p in a.get("plots", [])],
        "evidence": list((a.get("sub_hypothesis_evidence") or {}).keys()),
        "metrics_count": len(a.get("metrics", [])),
        "errors": res["errors"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if a.get("error"):
        print("分析层报错：", a["error"])
        return 1
    if a.get("task_type") != "multi_dataset" and (not a.get("metrics") or a.get("best_method") is None):
        print("分析未产生有效结果。")
        return 1
    return 0


def cmd_plan(args) -> int:
    """闭环：自然语言主题 → Planner 出 spec → 拉文献 → 跑分析 → 出论文。"""
    planner = _load_module(ROOT / "planner.py", "planner")
    literature = _load_module(ROOT / "literature.py", "literature")

    # 1) 数据集覆盖（用户自带数据 / 公开网址 / 指定内置集）
    data_override = None
    if args.data:
        data_override = {"source": "csv", "path": str(Path(args.data).resolve()),
                         "target": args.target or "target"}
        if args.sep:
            data_override["sep"] = args.sep
        if args.na_values:
            data_override["na_values"] = args.na_values
    elif args.url:
        data_override = {"source": "url", "url": args.url,
                         "target": args.target or "target"}
        if args.sep:
            data_override["sep"] = args.sep
        if args.na_values:
            data_override["na_values"] = args.na_values
    elif args.dataset:
        data_override = {"source": "sklearn", "name": args.dataset}

    # 2) Planner 出 spec
    client, model, extra = planner.llm_client()
    if client is None:
        print("[warn] 未检测到 API Key（.env 中 OPENAI_API_KEY），使用模板 spec 降级运行。")
    dataset_meta = _dataset_meta(data_override)
    pa = _load_module(ROOT / "agents" / "planning_agent.py", "planning_agent")
    spec = pa.run(args.topic, client, model, extra, data_override, dataset_meta=dataset_meta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[plan] 数据集={spec['data']}  任务={spec['task_type']}  "
          f"方法数={len(spec['candidate_methods'])}  指标={spec['metric']}")

    # 3) 文献（LM 驱动检索 Agent，无需 Key）
    if not args.no_literature:
        try:
            ga = _load_module(ROOT / "agents" / "literature_gather_agent.py", "lg_agent")
            _methods = [m.get("name") for m in (spec.get("candidate_methods") or [])
                        if isinstance(m, dict)]
            gres = ga.run(
                spec.get("literature_query") or args.topic,
                task_type=spec.get("task_type", "classification"),
                method_cn=_methods, dataset_query=spec.get("literature_query") or "",
                out_dir=out_dir, top_n=45)
            print(f"[GatherAgent] {'LM驱动' if gres['lm_driven'] else '规则兜底'}检索，"
                  f"精选 {len(gres['papers'])} 篇文献")
        except Exception as e:  # noqa: BLE001
            print(f"[literature] 检索失败（不影响主流程）：{e}")

    # 4) 跑分析 + 出论文
    res = run_task(spec, out_dir, make_paper=not args.no_paper, mode=args.mode)
    a = res["analysis"]
    summary = {
        "out_dir": res["out_dir"],
        "paper": res["paper"],
        "best_method": a.get("best_method"),
        "best_value": a.get("best_value"),
        "plots": [p.get("file") for p in a.get("plots", [])],
        "evidence": list((a.get("sub_hypothesis_evidence") or {}).keys()),
        "metrics_count": len(a.get("metrics", [])),
        "errors": res["errors"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if a.get("error"):
        print("分析层报错：", a["error"])
    return 0


def cmd_iterate(args) -> int:
    """Planner 迭代回路：plan → 跑 → 结果喂回 Planner → 扩宽（数据集/方法）→ 再跑，最多 N 轮。"""
    planner = _load_module(ROOT / "planner.py", "planner")
    literature = _load_module(ROOT / "literature.py", "literature")

    client, model, extra = planner.llm_client()
    data_override = None
    if args.data:
        data_override = {"source": "csv", "path": str(Path(args.data).resolve()),
                         "target": args.target or "target"}
        if args.sep:
            data_override["sep"] = args.sep
        if args.na_values:
            data_override["na_values"] = args.na_values
    elif args.url:
        data_override = {"source": "url", "url": args.url,
                         "target": args.target or "target"}
        if args.sep:
            data_override["sep"] = args.sep
        if args.na_values:
            data_override["na_values"] = args.na_values
    elif args.dataset:
        data_override = {"source": "sklearn", "name": args.dataset}

    if client is None:
        # 无 Key：退回单次 plan + 跑（不迭代）
        print("[warn] 未检测到 API Key，迭代回路不可用，执行单次 plan + 分析。")
        pa = _load_module(ROOT / "agents" / "planning_agent.py", "planning_agent")
        spec = pa.run(args.topic, None, None, None, data_override)
        out_dir = Path(args.out) / "round_1"
        out_dir.mkdir(parents=True, exist_ok=True)
        res = run_task(spec, out_dir, make_paper=not args.no_paper)
        print(json.dumps({"round": 1, "best": res["analysis"].get("best_method"),
                          "paper": res["paper"], "errors": res["errors"]},
                        ensure_ascii=False, indent=2))
        return 0

    pa = _load_module(ROOT / "agents" / "planning_agent.py", "planning_agent")
    prev_spec = pa.run(args.topic, client, model, extra, data_override)
    rounds = max(1, args.rounds)
    prev_result = None
    history = []
    for r in range(1, rounds + 1):
        out_dir = Path(args.out) / f"round_{r}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_literature and not (out_dir / "01_literature.json").exists():
            try:
                ga = _load_module(ROOT / "agents" / "literature_gather_agent.py", "lg_agent")
                _methods = [m.get("name") for m in (prev_spec.get("candidate_methods") or [])
                            if isinstance(m, dict)]
                ga.run(
                    prev_spec.get("literature_query") or args.topic,
                    task_type=prev_spec.get("task_type", "classification"),
                    method_cn=_methods, dataset_query=prev_spec.get("literature_query") or "",
                    out_dir=out_dir, top_n=45)
            except Exception as e:  # noqa: BLE001
                print(f"[literature] 检索失败（不影响主流程）：{e}")
        res = run_task(prev_spec, out_dir, make_paper=not args.no_paper, mode=args.mode)
        prev_result = res["analysis"]
        _ds_items = prev_spec.get("datasets") or []
        if not _ds_items and prev_spec.get("data"):
            _ds_items = [{"data": prev_spec["data"]}]
        ds_names = [(it.get("label") or (it.get("data") or it).get("name") or "?") for it in _ds_items]
        best = prev_result.get("best_method") or "(多数据集)"
        history.append({"round": r, "out_dir": str(out_dir), "best_method": best,
                        "paper": res["paper"], "errors": res["errors"]})
        print(f"[round {r}/{rounds}] 数据集={'/'.join(str(x) for x in ds_names)} 最优={best}")
        if r < rounds:
            nxt = planner.iterate_spec(prev_spec, prev_result, client, model, extra, r + 1)
            if not isinstance(nxt, dict):
                print("[warn] 迭代规划失败，终止回路。")
                break
            prev_spec = nxt

    (Path(args.out) / "iteration_summary.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== 迭代总览 ===")
    print(json.dumps(history, ensure_ascii=False, indent=2))
    return 0


def cmd_datasets(args) -> int:
    """列出可执行数据集白名单：学生据此判断自己的题目能否直接跑通。"""
    import engine.topic_parser as tp

    rows = []
    for ds in tp.DATASET_CATALOG:
        rows.append({
            "key": ds["key"],
            "label": ds.get("label"),
            "source": ds.get("source"),
            "task": ds.get("task"),
            "modality": ds.get("modality", "tabular"),
            "scale": (f"{ds.get('n')}×{ds.get('feat')}" if ds.get("n")
                      else (f"OpenML#{ds.get('dataset_id')}" if ds.get("dataset_id") else "按实际加载")),
            "keywords": "、".join((ds.get("domain_kw") or [])[:6]),
        })
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    print(f"{'key':20s} {'模态':6s} {'任务':12s} {'规模':14s} 说明")
    print("-" * 96)
    for r in rows:
        print(f"{r['key']:20s} {r['modality']:6s} {r['task']:12s} {r['scale']:14s} {r['label']}")
        print(f"{'':20s} 关键词：{r['keywords']}")
    print(f"\n共 {len(rows)} 个可直接执行的数据集。题目里出现「无匹配」时，"
          f"用 --data/--dataset 指定自有数据，或在此目录补充新条目。")
    return 0


def cmd_from_title(args) -> int:
    """题目驱动执行：老师给题目/方向 → 确定性解析 → 匹配数据集 → 出论文。

    不依赖 LLM（纯关键词规则），断网也能跑。论文标题直接使用老师给的题目。
    """
    import engine.topic_parser as tp  # 同源模块，直接 import 即可

    title = args.title.strip()
    direction = (args.direction or "").strip()
    data_override = None
    if args.data:
        data_override = {"source": "csv", "path": str(Path(args.data).resolve()),
                         "target": args.target or "target"}
        if args.sep:
            data_override["sep"] = args.sep
        if args.na_values:
            data_override["na_values"] = args.na_values
    elif args.dataset:
        data_override = {"source": "sklearn", "name": args.dataset}

    method_hints = None
    if args.methods:
        method_hints = [m.strip() for m in args.methods.split(",") if m.strip()]

    # 解析（先不绑定数据，便于在缺失时给出可读提示）
    intent = tp.parse_title(title, direction)
    print(f"[解析] 任务类型={intent['task_type']}  匹配数据集={intent['matched_dataset'] or '无'}  "
          f"方法提示={intent['method_hints'] or '默认'}  "
          f"暂不支持={intent['unsupported_methods'] or '无'}")

    if not data_override:
        matched = tp.match_dataset(title, direction)
        if not matched:
            print("[错误] 题目未匹配到本地数据集。可用数据集：")
            for ds in tp.DATASET_CATALOG:
                print(f"   - {ds['label']}（{ds['task']}）：关键词 {ds['domain_kw'][:3]}...")
            print("请用 --data <CSV路径> --target <目标列> 自带数据，或 --dataset <sklearn名> 指定内置集。")
            return 1

    spec = tp.assemble_title_spec(title, direction, data_override=data_override,
                                 method_hints=method_hints)
    if spec.get("data") is None:
        print("[错误] 无法确定数据集绑定。")
        return 1
    print(f"[绑定] {spec['_match_note']}")
    if spec.get("_unsupported_methods"):
        print(f"[提示] 题目提及但引擎暂未内置的方法：{spec['_unsupported_methods']}；"
              f"将以经典机器学习方法 + 融合集成方法完成，论文如实说明。")

    # 预演模式：只打印「会跑什么」，不落盘、不打网关，避免错配题白跑几分钟
    if getattr(args, "dry_run", False):
        _d = spec.get("data") or {}
        print("\n[预演] 本任务将执行：")
        print(f"  论文题目：{spec.get('title')}")
        print(f"  任务类型：{spec.get('task_type')}（主指标 {spec.get('metric')}，{spec.get('objective')}）")
        print(f"  数据集：{spec.get('_match_note')}")
        print(f"  数据绑定：source={_d.get('source')} "
              f"{'name=' + str(_d.get('name')) if _d.get('name') else ''}"
              f"{'data_id=' + str(_d.get('dataset_id')) if _d.get('dataset_id') else ''}"
              f"{'path=' + str(_d.get('path')) if _d.get('path') else ''}")
        print(f"  候选方法（{len(spec.get('candidate_methods') or [])} 个）："
              f"{'、'.join(m.get('name', '?') for m in (spec.get('candidate_methods') or []))}")
        print(f"  是否要求方法贡献：{'是' if spec.get('_needs_contribution') else '否（忠实复现题目点名方法）'}")
        print(f"  文献检索式：{spec.get('literature_query')}")
        print(f"  研究问题：{spec.get('research_question')}")
        print("\n未执行。去掉 --dry-run 即开始正式运行。")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_task(spec, out_dir, make_paper=not args.no_paper, mode=args.mode,
                   force=getattr(args, "force", False))
    a = res["analysis"]
    summary = {
        "out_dir": res["out_dir"],
        "paper": res["paper"],
        "best_method": a.get("best_method"),
        "best_value": a.get("best_value"),
        "plots": [p.get("file") for p in a.get("plots", [])],
        "errors": res["errors"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if a.get("error"):
        print("分析层报错：", a["error"])
        return 1
    if a.get("task_type") != "multi_dataset" and (not a.get("metrics") or a.get("best_method") is None):
        print("分析未产生有效结果。")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="通用 Auto Research 入口：agent 喂 spec，它跑分析并产出论文"
    )
    sub = ap.add_subparsers(dest="cmd")

    rp = sub.add_parser("run", help="运行一份任务 spec 端到端产出论文")
    rp.add_argument("--spec", required=True, help="任务 spec JSON 路径")
    rp.add_argument("--out", default="out_auto", help="产物输出目录")
    rp.add_argument("--no-paper", action="store_true", help="只跑分析，不出论文")
    rp.add_argument("--mode", default="general",
                    help="机构模式：general（默认，通用学术标准）或 sztu（深圳技术大学预设）")
    rp.set_defaults(func=cmd_run)

    pp = sub.add_parser("plan", help="给一句研究主题，自动规划 spec + 拉文献 + 出论文")
    pp.add_argument("--topic", required=True, help="自然语言研究主题")
    pp.add_argument("--out", default="out_plan", help="产物输出目录")
    pp.add_argument("--dataset", default="", help="指定内置 sklearn 数据集名（覆盖 Planner 选择）")
    pp.add_argument("--data", default="", help="自带 CSV 路径（与 --target 配合）")
    pp.add_argument("--url", default="", help="公开数据集网址（CSV/XLSX，与 --target 配合）")
    pp.add_argument("--target", default="", help="CSV 中目标列名或列索引")
    pp.add_argument("--sep", default="", help="CSV 分隔符（如分号 ';'，默认逗号）")
    pp.add_argument("--na_values", default="", help="缺失值标记（如 '?'，多个用逗号分隔）")
    pp.add_argument("--no-literature", action="store_true", help="跳过 arXiv 文献检索")
    pp.add_argument("--no-paper", action="store_true", help="只跑分析，不出论文")
    pp.add_argument("--mode", default="general",
                    help="机构模式：general（默认，通用学术标准）或 sztu（深圳技术大学预设）")
    pp.set_defaults(func=cmd_plan)

    ip = sub.add_parser("iterate", help="给主题，Planner 多轮迭代（扩宽数据集/方法）自动出论文")
    ip.add_argument("--topic", required=True, help="自然语言研究主题")
    ip.add_argument("--out", default="out_iterate", help="产物输出根目录（下分 round_1..N）")
    ip.add_argument("--rounds", type=int, default=3, help="迭代轮数（默认 3）")
    ip.add_argument("--dataset", default="", help="指定内置 sklearn 数据集名（覆盖 Planner 首轮选择）")
    ip.add_argument("--data", default="", help="自带 CSV 路径（与 --target 配合）")
    ip.add_argument("--url", default="", help="公开数据集网址（CSV/XLSX，与 --target 配合）")
    ip.add_argument("--target", default="", help="CSV 中目标列名或列索引")
    ip.add_argument("--sep", default="", help="CSV 分隔符（如分号 ';'，默认逗号）")
    ip.add_argument("--na_values", default="", help="缺失值标记（如 '?'，多个用逗号分隔）")
    ip.add_argument("--no-literature", action="store_true", help="跳过 arXiv 文献检索")
    ip.add_argument("--no-paper", action="store_true", help="只跑分析，不出论文")
    ip.add_argument("--mode", default="general",
                    help="机构模式：general（默认，通用学术标准）或 sztu（深圳技术大学预设）")
    ip.set_defaults(func=cmd_iterate)

    fp = sub.add_parser("from_title", help="给老师给的论文题目/方向，确定性解析并出论文（不依赖 LLM）")
    fp.add_argument("--title", required=True, help="老师给的论文题目（如「基于随机森林的心脏病预测研究」）")
    fp.add_argument("--direction", default="", help="导师给的大方向/补充说明（可选，辅助领域匹配）")
    fp.add_argument("--out", default="out_title", help="产物输出目录")
    fp.add_argument("--data", default="", help="自带 CSV 路径（与 --target 配合，覆盖自动匹配）")
    fp.add_argument("--dataset", default="", help="指定 sklearn 内置数据集名（如 iris / wine / breast_cancer）")
    fp.add_argument("--target", default="", help="CSV 中目标列名或列索引")
    fp.add_argument("--sep", default="", help="CSV 分隔符（如分号 ';'）")
    fp.add_argument("--na_values", default="", help="缺失值标记（如 '?'）")
    fp.add_argument("--methods", default="", help="指定方法 key 列表（rf,lr,svm,knn,gb,dt,nb，逗号分隔）")
    fp.add_argument("--no-paper", action="store_true", help="只跑分析，不出论文")
    fp.add_argument("--force", action="store_true",
                    help="强制重抓文献/重跑分析（默认断点续跑，复用已有 01_literature.json）")
    fp.add_argument("--dry-run", action="store_true",
                    help="只预演解析结果（数据集/方法/检索式），不跑实验不出论文")
    fp.add_argument("--mode", default="general",
                    help="机构模式：general（默认，通用学术标准）或 sztu（深圳技术大学预设）")
    fp.set_defaults(func=cmd_from_title)

    dp = sub.add_parser("datasets", help="列出当前可直接跑通的数据集目录（题目匹配用的白名单）")
    dp.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本消费")
    dp.set_defaults(func=cmd_datasets)

    args = ap.parse_args()
    if getattr(args, "func", None) is None:
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
