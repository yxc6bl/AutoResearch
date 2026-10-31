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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS_ENGINE = ROOT / "engine" / "analysis_engine.py"
GEN_PAPER = ROOT / "paper" / "generate_paper.py"


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
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def run_task(spec: dict, out_dir, make_paper: bool = True) -> dict:
    """程序化 API：给定 spec，端到端产出分析 + 论文。

    返回 dict：{analysis, out_dir, paper(路径或None), errors(list)}
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 写 spec 副本（论文生成器要读标题 / 指标 / 假设等）
    (out_dir / "task_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2) 文献：已存在真实文献则保留；否则尝试从 arXiv 抓取真实文献
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
    if _need_fetch:
        try:
            la = _load_module(ROOT / "agents" / "literature_agent.py", "literature_agent")
            _q = (spec.get("literature_query") or spec.get("topic")
                  or spec.get("direction") or spec.get("research_question") or "")
            if _q:
                _papers = la.run(_q, max_results=10, out_path=lit_path,
                                 source_tag=str(out_dir))
                print(f"[literature] 检索/记忆到 {len(_papers)} 篇文献")
            else:
                lit_path.write_text(json.dumps({"papers": []}, ensure_ascii=False), encoding="utf-8")
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

    # 3) 确定性分析（数据集与方法全部从 spec 解析）
    exp_path = out_dir / "03_experiment.json"
    mod_spec = importlib.util.spec_from_file_location("analysis_engine", ANALYSIS_ENGINE)
    ae = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(ae)
    # 只有 ≥2 个有效数据集才进入多数据集模式。
    # 若规划器只给出 1 个数据集（或列表在校验后被过滤到只剩 1 个），仍按单数据集处理，
    # 否则 run_multi_analysis 会产出只有一行的「跨数据集对比表」，反而制造新的虚假表述。
    _ds_list = spec.get("datasets") or spec.get("data_list") or []
    if _ds_list and len(_ds_list) >= 2:
        # 多数据集对比：run_multi_analysis 内部写 per-dataset 结果 + combined
        analysis = ae.run_multi_analysis(spec, exp_path)
    else:
        analysis = ae.run_analysis(spec, exp_path)
        # run_analysis 返回的就是扁平 ANALYSIS_JSON，直接落盘
        exp_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    result: dict = {"analysis": analysis, "out_dir": str(out_dir), "paper": None, "errors": []}
    if analysis.get("error"):
        result["errors"].append("分析失败: " + str(analysis["error"]))
        return result

    # 4) 生成论文（失败不影响分析产物）
    if make_paper:
        try:
            proc = subprocess.run(
                [
                    sys.executable, str(GEN_PAPER),
                    "--output-dir", str(out_dir),
                    "--task-spec", str(out_dir / "task_spec.json"),
                ],
                capture_output=True, text=True, cwd=str(ROOT), timeout=600,
                env=_env_with_dotenv(),
            )
            if proc.returncode != 0:
                result["errors"].append("generate_paper 失败: " + (proc.stderr or "")[-2500:])
            else:
                paper = out_dir / "06_paper" / "paper_full.docx"
                result["paper"] = str(paper) if paper.exists() else None
                # 5) 质量评分门（EvalAgent）：评判文献质量 + 论文质量，给出 verdict
                try:
                    ea = _load_module(ROOT / "agents" / "eval_agent.py", "eval_agent")
                    _md = out_dir / "06_paper" / "paper_full.md"
                    _litc = out_dir / "02_literature_candidates.json"
                    _cands = None
                    if _litc.exists():
                        try:
                            _d = json.loads(_litc.read_text(encoding="utf-8"))
                            _cands = _d if isinstance(_d, list) else (_d.get("papers") or [])
                        except Exception:
                            _cands = None
                    if _md.exists():
                        _verdict = ea.run(
                            str(_md), lit_candidates=_cands,
                            out_path=out_dir / "06_paper" / "07_eval_verdict.json")
                        result["eval"] = _verdict
                        _ok = "[通过]" if _verdict["passed"] else "[未通过(建议重做)]"
                        print(f"[EvalAgent] 质量评分门：{_ok} | 综合 {_verdict['overall']} | "
                              f"硬门 {_verdict['gate']} | 文献 {_verdict.get('literature_grade')}")
                        if not _verdict["passed"]:
                            print(f"[EvalAgent] 主要问题：{'; '.join(_verdict['issues'][:5])}")
                    else:
                        print("[EvalAgent] 未找到 paper_full.md，跳过评分")
                except Exception as e:  # noqa: BLE001
                    print(f"[EvalAgent] 评分失败（不影响主流程）：{e}")
        except Exception as e:  # noqa: BLE001
            result["errors"].append(f"generate_paper 异常: {e}")

    return result


def cmd_run(args) -> int:
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    res = run_task(spec, args.out, make_paper=not args.no_paper)
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

    # 1) 数据集覆盖（用户自带数据 / 指定内置集）
    data_override = None
    if args.data:
        data_override = {"source": "csv", "path": str(Path(args.data).resolve()),
                         "target": args.target or "target"}
    elif args.dataset:
        data_override = {"source": "sklearn", "name": args.dataset}

    # 2) Planner 出 spec
    client, model, extra = planner.llm_client()
    if client is None:
        print("[warn] 未检测到 API Key（.env 中 OPENAI_API_KEY），使用模板 spec 降级运行。")
    spec = planner.plan_spec(args.topic, client, model, extra, data_override)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[plan] 数据集={spec['data']}  任务={spec['task_type']}  "
          f"方法数={len(spec['candidate_methods'])}  指标={spec['metric']}")

    # 3) 文献（arXiv，无需 Key）
    if not args.no_literature:
        try:
            la = _load_module(ROOT / "agents" / "literature_agent.py", "literature_agent")
            papers = la.run(
                spec.get("literature_query") or args.topic,
                max_results=10, out_path=out_dir / "01_literature.json",
                source_tag=str(out_dir),
            )
            print(f"[literature] 检索/记忆到 {len(papers)} 篇文献")
        except Exception as e:  # noqa: BLE001
            print(f"[literature] 检索失败（不影响主流程）：{e}")

    # 4) 跑分析 + 出论文
    res = run_task(spec, out_dir, make_paper=not args.no_paper)
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
    elif args.dataset:
        data_override = {"source": "sklearn", "name": args.dataset}

    if client is None:
        # 无 Key：退回单次 plan + 跑（不迭代）
        print("[warn] 未检测到 API Key，迭代回路不可用，执行单次 plan + 分析。")
        spec = planner.plan_spec(args.topic, None, None, None, data_override)
        out_dir = Path(args.out) / "round_1"
        out_dir.mkdir(parents=True, exist_ok=True)
        res = run_task(spec, out_dir, make_paper=not args.no_paper)
        print(json.dumps({"round": 1, "best": res["analysis"].get("best_method"),
                          "paper": res["paper"], "errors": res["errors"]},
                        ensure_ascii=False, indent=2))
        return 0

    prev_spec = planner.plan_spec(args.topic, client, model, extra, data_override)
    rounds = max(1, args.rounds)
    prev_result = None
    history = []
    for r in range(1, rounds + 1):
        out_dir = Path(args.out) / f"round_{r}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_literature and not (out_dir / "01_literature.json").exists():
            try:
                la = _load_module(ROOT / "agents" / "literature_agent.py", "literature_agent")
                la.run(
                    prev_spec.get("literature_query") or args.topic,
                    max_results=10, out_path=out_dir / "01_literature.json",
                    source_tag=str(out_dir),
                )
            except Exception as e:  # noqa: BLE001
                print(f"[literature] 检索失败（不影响主流程）：{e}")
        res = run_task(prev_spec, out_dir, make_paper=not args.no_paper)
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="通用 Auto Research 入口：agent 喂 spec，它跑分析并产出论文"
    )
    sub = ap.add_subparsers(dest="cmd")

    rp = sub.add_parser("run", help="运行一份任务 spec 端到端产出论文")
    rp.add_argument("--spec", required=True, help="任务 spec JSON 路径")
    rp.add_argument("--out", default="out_auto", help="产物输出目录")
    rp.add_argument("--no-paper", action="store_true", help="只跑分析，不出论文")
    rp.set_defaults(func=cmd_run)

    pp = sub.add_parser("plan", help="给一句研究主题，自动规划 spec + 拉文献 + 出论文")
    pp.add_argument("--topic", required=True, help="自然语言研究主题")
    pp.add_argument("--out", default="out_plan", help="产物输出目录")
    pp.add_argument("--dataset", default="", help="指定内置 sklearn 数据集名（覆盖 Planner 选择）")
    pp.add_argument("--data", default="", help="自带 CSV 路径（与 --target 配合）")
    pp.add_argument("--target", default="", help="CSV 中目标列名或列索引")
    pp.add_argument("--no-literature", action="store_true", help="跳过 arXiv 文献检索")
    pp.add_argument("--no-paper", action="store_true", help="只跑分析，不出论文")
    pp.set_defaults(func=cmd_plan)

    ip = sub.add_parser("iterate", help="给主题，Planner 多轮迭代（扩宽数据集/方法）自动出论文")
    ip.add_argument("--topic", required=True, help="自然语言研究主题")
    ip.add_argument("--out", default="out_iterate", help="产物输出根目录（下分 round_1..N）")
    ip.add_argument("--rounds", type=int, default=3, help="迭代轮数（默认 3）")
    ip.add_argument("--dataset", default="", help="指定内置 sklearn 数据集名（覆盖 Planner 首轮选择）")
    ip.add_argument("--data", default="", help="自带 CSV 路径（与 --target 配合）")
    ip.add_argument("--target", default="", help="CSV 中目标列名或列索引")
    ip.add_argument("--no-literature", action="store_true", help="跳过 arXiv 文献检索")
    ip.add_argument("--no-paper", action="store_true", help="只跑分析，不出论文")
    ip.set_defaults(func=cmd_iterate)

    args = ap.parse_args()
    if getattr(args, "func", None) is None:
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
