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
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from engine import gates, trace  # noqa: E402

try:  # 论文类型注册表（v3 可插拔骨架）
    from paper.thesis_template import (
        is_experimental as _is_exp,
        paper_type_registry as _paper_types,
    )
    PAPER_TYPE_CHOICES = list(_paper_types().keys())
except Exception:  # noqa: BLE001
    def _is_exp(pt):
        return pt in (None, "", "method_comparison")
    PAPER_TYPE_CHOICES = ["method_comparison"]

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


def wait_for_llm(max_wait: int = 0, interval: int = 60) -> bool:
    """等待模型恢复：模型不可用时每隔 interval 秒重试一次，最多等 max_wait 秒。

    用户的原话是「网关卡了，不就可以等待吗」——等待比降级更符合「只要高质量产出」
    的定位。这里只做有上限的等待，不做无限重试。
    """
    from engine import llm_client as LLM

    if LLM.available():
        return True
    if max_wait <= 0:
        return False
    waited = 0
    while waited < max_wait:
        step = min(interval, max_wait - waited)
        print(f"[等待] 模型不可用，{step}s 后重试（已等待 {waited}s / 上限 {max_wait}s）")
        time.sleep(step)
        waited += step
        LLM.reset_cache()
        if LLM.available():
            print(f"[等待] 模型已恢复（等待 {waited}s）")
            return True
    print(f"[等待] 超过上限 {max_wait}s，模型仍不可用")
    return False


def _lit_top_n() -> int:
    """文献精选上限（可配 LIT_TOP_N，默认 60）。

    是「最多」不是「目标」：实际篇数由角色配额与质量线决定，
    select_by_role 报告里的 n_eligible / capped_by_cap 会显示是否被上限截断。
    """
    try:
        return max(10, int(os.environ.get("LIT_TOP_N", "60")))
    except ValueError:
        return 60


def _existing_artifacts(out_dir: Path) -> list[str]:
    """列出目录里已有的产物，用于失败时告诉用户「哪些东西保住了」。"""
    names = ["task_spec.json", "01_literature.json", "02_hypotheses.json",
             "03_experiment.json", "04_review.json", "06_paper/paper_full.md"]
    return [n for n in names if (out_dir / n).exists()]


def mark_delivery(out_dir, llm_written=None) -> dict:
    """判定这次产物能不能当交付稿，并写 06_paper/DELIVERY_STATUS.json。

    规则很直白：模型没参与写作（llm_written=False 或正文中文不足 6000 字）→ 判为
    「不可交付」。这类稿子只能用于自测/定位/演示链路，不能当成品交上去。
    """
    out_dir = Path(out_dir)
    md = out_dir / "06_paper" / "paper_full.md"
    # 生成器单独落盘的写作参与证据（比 stdout 反解可靠）；调用方没显式传值时就地补读。
    _wstat: dict = {}
    _ws_path = out_dir / "06_paper" / "WRITE_STATUS.json"
    if _ws_path.exists():
        try:
            _wstat = json.loads(_ws_path.read_text(encoding="utf-8"))
        except Exception:
            _wstat = {}
    if llm_written is None and _wstat.get("llm_written") is not None:
        llm_written = _wstat.get("llm_written")
    n_chars = 0
    if md.exists():
        n_chars = len(re.findall(r"[\u4e00-\u9fff]", md.read_text(encoding="utf-8")))
    degraded = (llm_written is False) or (n_chars and n_chars < 6000) or (n_chars == 0)
    reasons = []
    if llm_written is False:
        reasons.append("模型未参与写作（生成器走了确定性模板骨架）")
    if n_chars and n_chars < 6000:
        reasons.append(f"正文仅 {n_chars} 个汉字（正常应为 1.2 万字以上）")
    if n_chars == 0:
        reasons.append("未找到正文")
    status = {
        "deliverable": not degraded,
        "llm_written": llm_written,
        "llm_sections": _wstat.get("llm_sections") or [],
        "write_model": _wstat.get("model") or "",
        "chinese_chars": n_chars,
        "reasons": reasons,
        "hint": ("可交付" if not degraded else
                 "不可交付：仅用于自测/定位链路，请等模型可用后重跑写作"),
    }
    try:
        target = out_dir / "06_paper" / "DELIVERY_STATUS.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    if degraded:
        print("【不可交付】" + "；".join(reasons))
        print("           该产物仅供自测/定位，不要作为成品提交。")
    else:
        print(f"【可交付】正文 {n_chars} 字，模型参与写作。")
    return status


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
    _t_stage = time.time()
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
    # 复用缓存文献时补做一次审评：否则 G08 永远是 unknown，覆盖缺口/冗余没人看，
    # 「复用」变成「不检查」。审评只看已有清单，不重新联网抓取。
    if lit_path.exists() and not (out_dir / "04_literature_review.json").exists():
        try:
            _lp = json.loads(lit_path.read_text(encoding="utf-8"))
            _papers = _lp.get("papers") or []
            if _papers:
                ra = _load_module(ROOT / "agents" / "literature_review_agent.py", "lr_agent")
                _topic = (spec.get("title") or spec.get("topic") or spec.get("direction")
                          or spec.get("research_question") or spec.get("literature_query") or "")
                try:
                    from engine import selection_ledger as SL
                    _prefs = SL.preference_prompt(_topic)
                except Exception:
                    _prefs = ""
                _lrev = ra.run(_topic, _papers, _lp.get("candidates"), _prefs)
                (out_dir / "04_literature_review.json").write_text(
                    json.dumps(_lrev, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[ReviewAgent] 补审缓存文献：覆盖 {_lrev.get('coverage_score')}/10，"
                      f"缺口 {len(_lrev.get('missing_themes') or [])} 项")
                try:
                    from engine import selection_ledger as SL
                    SL.attach_review(out_dir, _lrev)
                except Exception:
                    pass
        except Exception as e:  # noqa: BLE001
            print(f"[ReviewAgent] 缓存文献审评跳过：{e}")
    if _need_fetch:
        try:
            ga = _load_module(ROOT / "agents" / "literature_gather_agent.py", "lg_agent")
            _topic = (spec.get("title") or spec.get("topic") or spec.get("direction")
                      or spec.get("research_question") or spec.get("literature_query") or "")
            _methods = [m.get("name") for m in (spec.get("candidate_methods") or [])
                        if isinstance(m, dict)]
            _dq = spec.get("literature_query") or ""
            _gres = ga.run(_topic, task_type=spec.get("task_type", "classification"),
                           method_cn=_methods, dataset_query=_dq, out_dir=out_dir, top_n=_lit_top_n())
            print(f"[GatherAgent] {'LM驱动' if _gres['lm_driven'] else '规则兜底'}检索，"
                  f"精选 {len(_gres['papers'])} 篇文献")
            # 第二双眼睛：文献审评 Agent，独立判断覆盖度/冗余/缺口
            try:
                ra = _load_module(ROOT / "agents" / "literature_review_agent.py", "lr_agent")
                _cands = _gres.get("candidates") or []
                try:
                    from engine import selection_ledger as SL
                    _prefs = SL.preference_prompt(_topic)
                except Exception:
                    _prefs = ""
                _lrev = ra.run(_topic, _gres["papers"], _cands, _prefs)
                (out_dir / "04_literature_review.json").write_text(
                    json.dumps(_lrev, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[ReviewAgent] 文献覆盖评分 {_lrev.get('coverage_score')}/10"
                      f"（{'LM' if _lrev.get('lm_driven') else '规则'}）；"
                      f"缺口：{_lrev.get('missing_themes') or '无'}")
                try:
                    from engine import feedback as FB
                    _nfb = FB.record_literature_review(_lrev, str(out_dir), _topic)
                    if _nfb:
                        print(f"[ReviewAgent] 已记录 {_nfb} 条覆盖缺口 → memory/feedback.jsonl")
                except Exception as e:  # noqa: BLE001
                    print(f"[ReviewAgent] 缺口入账跳过：{e}")
                try:
                    from engine import selection_ledger as SL
                    SL.attach_review(out_dir, _lrev)
                    # 审评给出的补强检索式存进记忆，下一轮检索计划会并入；
                    # 本轮先走原有的即时补抓，两条路不冲突。
                    _nq = SL.add_feedback_queries(
                        _lrev.get("recommended_add_queries") or [],
                        source="literature_review",
                        note="审评缺口：" + "；".join(
                            map(str, (_lrev.get("missing_themes") or [])[:3])))
                    if _nq:
                        print(f"[检索规划] 审评给出 {_nq} 条补强检索式，已存记忆供下轮并入")
                except Exception:
                    pass
                # 审评 → 补抓闭环：缺口不再只写进 04 文件就结束，
                # 而是转成检索式再抓一轮，新增文献同样逐篇读，最后覆盖 01_literature.json。
                # 想省时间可设 LIT_REFETCH=0 关闭。
                if os.environ.get("LIT_REFETCH", "1") != "0":
                    try:
                        from agents import literature_refetch_agent as LRA
                        _rr = LRA.run(_topic, out_dir, _lrev, _gres["papers"],
                                      budget_sec=60, top_n=_lit_top_n())
                        if _rr.get("status") == "ok":
                            print(f"[RefetchAgent] 文献集已更新为 {_rr['n_after']} 篇"
                                  f"（LM 阅读覆盖 {_rr['read_coverage_after']:.0%}）")
                            try:
                                from engine import lit_plan as LP
                                _pl = LP.load(out_dir)
                                if _pl:
                                    LP.add_round(
                                        _pl, "refetch",
                                        _lrev.get("recommended_add_queries") or [],
                                        int(_rr.get("n_after", 0)) - int(_rr.get("n_before", 0)),
                                        "审评缺口即时补抓")
                                    LP.write(out_dir, _pl)
                            except Exception:
                                pass
                    except Exception as e:  # noqa: BLE001
                        print(f"[RefetchAgent] 补抓跳过（不影响主流程）：{e}")
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

    trace.record(out_dir, "literature", time.time() - _t_stage,
                 detail=("复用已有" if not _need_fetch else "新检索"))
    _t_stage = time.time()

    # 3) 确定性分析（ExperimentAgent 封装，写 project memory）
    #    非实验型论文（survey / system_design）跳过实验引擎，写一份标记 paper_type 的桩
    #    产物，使下游闸门/评审按类型豁免「实验有效」硬门，且不依赖实验事实做零编造门。
    paper_type = spec.get("paper_type", "method_comparison")
    result: dict = {"analysis": None, "out_dir": str(out_dir), "paper": None, "errors": []}
    exp_path = out_dir / "03_experiment.json"
    if not _is_exp(paper_type):
        analysis = {
            "paper_type": paper_type,
            "task_type": paper_type,
            "note": f"非实验型论文（{paper_type}），不跑确定性实验引擎",
            "analysis": {"seed_variance": {}},
            "metrics": [],
            "best_method": None,
        }
        try:
            exp_path.parent.mkdir(parents=True, exist_ok=True)
            exp_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            result["errors"].append("实验桩写入失败: " + str(e))
        analysis = analysis  # 下游统一按 dict 使用
    else:
        try:
            ea = _load_module(ROOT / "agents" / "experiment_agent.py", "experiment_agent")
            analysis = ea.run(spec, exp_path)
        except Exception as e:  # noqa: BLE001
            result["errors"].append("实验分析失败: " + str(e))
            return result

        if analysis.get("error"):
            result["errors"].append("分析失败: " + str(analysis["error"]))
            return result

    result = {"analysis": analysis, "out_dir": str(out_dir), "paper": None, "errors": []}

    trace.record(out_dir, "analysis", time.time() - _t_stage,
                 status=("error" if result["analysis"].get("error") else "ok"),
                 detail=str(result["analysis"].get("best_method") or ""))
    _t_stage = time.time()

    # 4) 生成论文 + 评审闭环（WritingAgent 写作 → EvalAgent 的 LLM 评审官审 → 回滚重写）
    #    一个人写、另一个人审：LLM 评审官审不过（verdict=revise）就把意见写成 04_review.json，
    #    写作端下一轮带意见重写，最多 _MAX_ROUNDS 轮，取 LLM 评审分最高的一版。网关不可用时
    #    LLM 评审官不可用，自动退化为单轮规则评分（旧行为），不阻塞主流程。
    _llm_written = None
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
            _rounds = []
            _last_round = 0
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
                # 写作端 stdout 里带 llm_written，必须透传给交付判定：
                # 以前 run_task 调 mark_delivery(out_dir) 不传，DELIVERY_STATUS 里
                # 这个字段永远是 null，严格模式只能把它判成「未验证」。
                if _wres.get("llm_written") is not None:
                    _llm_written = _wres.get("llm_written")
                if not _md.exists():
                    break
                _last_round = _rnd + 1
                # 每轮留稿 + 留评审结论：此前「取最优」只记了分数，文件早被下一轮覆盖，
                # 所谓回滚闭环根本拿不回最优那一版。
                _snap = out_dir / "06_paper" / f"paper_full_round{_rnd + 1}.md"
                _vpath = out_dir / "06_paper" / f"07_eval_verdict_round{_rnd + 1}.json"
                try:
                    shutil.copy2(_md, _snap)
                except Exception:
                    _snap = None
                # LLM 评审官（含确定性硬门）打分
                _verdict = ea.run(str(_md), lit_candidates=_cands, mode=mode,
                                 out_path=_vpath)
                try:
                    shutil.copy2(_vpath, out_dir / "06_paper" / "07_eval_verdict.json")
                except Exception:
                    pass
                result["eval"] = _verdict
                _lj = _verdict.get("llm_judge")
                _ok = "[通过]" if _verdict["passed"] else "[未通过(建议重做)]"
                _ljs = (f" | LLM评审 {_lj['overall']}" if (_lj and _lj.get("available")) else "")
                print(f"[EvalAgent] 质量评分门：{_ok} | 综合 {_verdict['overall']} | "
                      f"硬门 {_verdict['gate']} | 文献 {_verdict.get('literature_grade')}{_ljs}")
                # 记录最优（优先 LLM 评审分，无 LLM 时用规则综合分；同分时幻觉少者优先）
                _score = (_lj.get("overall") if (_lj and _lj.get("available")) else _verdict["overall"])
                _n_hall = (len(_lj.get("hallucinations") or [])
                           if (_lj and _lj.get("available")) else 0)
                _key = (_score, -_n_hall)
                _rounds.append({"round": _rnd + 1, "score": _score,
                                "hallucinations": _n_hall, "passed": bool(_verdict["passed"]),
                                "snapshot": _snap.name if _snap else ""})
                if _best is None or _key > _best["key"]:
                    _best = {"score": _score, "round": _rnd + 1, "key": _key,
                             "snapshot": str(_snap) if _snap else ""}
                # 评审官要求修订且还有轮次 → 写回滚简报，下一轮带意见重写
                if (_rnd < _MAX_ROUNDS - 1 and _lj and _lj.get("available")
                        and _lj.get("verdict") == "revise"):
                    _review_brief.write_text(json.dumps({
                        # 幻觉清单必须交给写作端：它是逐条对照实验事实抓出的矛盾，
                        # 只给 suggestions/unsupported_claims 会让下一轮重写漏掉真问题。
                        "hallucinations": _lj.get("hallucinations", []),
                        "unsupported_claims": _lj.get("unsupported_claims", []),
                        "suggestions": _lj.get("suggestions", []),
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"[EvalAgent] LLM评审官建议修订，回滚重写第 {_rnd + 2} 轮…")
                    continue
                break
            if _verdict is not None and not _verdict["passed"]:
                print(f"[EvalAgent] 主要问题：{'; '.join(_verdict['issues'][:5])}")
            # 主流水线此前只有 step eval 会入账，run/from_title 的评审意见全丢了。
            # 这里按最终 verdict 记一次（含 LLM 评审官的幻觉与无支撑断言，带来源）。
            if _verdict is not None:
                try:
                    from engine import feedback as FB
                    _topic0 = str(spec.get("title") or spec.get("topic") or "")
                    _n = FB.record_eval_verdict(_verdict, str(out_dir), _topic0)
                    if _n:
                        print(f"[反馈台账] 已记录 {_n} 条评审意见 → memory/feedback.jsonl")
                    # 销账：同题以前留下的问题，在这一轮过了硬门且零幻觉之后才算真修好。
                    _lj0 = _verdict.get("llm_judge") or {}
                    if (_verdict.get("gate") == "hard_pass"
                            and not (_lj0.get("hallucinations") or [])):
                        _closed = FB.close_for_topic(
                            _topic0, str(out_dir),
                            reason="同题后续运行 hard_pass 且幻觉为 0")
                        if _closed:
                            print(f"[反馈台账] 同题历史问题已销账 {_closed} 条")
                except Exception as e:  # noqa: BLE001
                    print(f"[反馈台账] 记录跳过：{e}")
            if _rounds:
                try:
                    (out_dir / "06_paper" / "REVIEW_ROUNDS.json").write_text(json.dumps({
                        "rounds": _rounds,                         "best": _best, "last_round": _last_round,
                        "note": "若最优不是最后一轮，paper_full.md、07_eval_verdict.json 与 "
                                "paper_full.docx 均已回滚并重转到最优轮，三者同源。",
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
            if _best is not None:
                print(f"[EvalAgent] 评审闭环完成：最优为第 {_best['round']} 轮（评分 {_best['score']}）")
                if _best["round"] != _last_round and _best.get("snapshot"):
                    try:
                        shutil.copy2(_best["snapshot"], _md)
                        _vb = out_dir / "06_paper" / f"07_eval_verdict_round{_best['round']}.json"
                        if _vb.exists():
                            shutil.copy2(_vb, out_dir / "06_paper" / "07_eval_verdict.json")
                        print(f"[EvalAgent] 已回滚到最优第 {_best['round']} 轮稿件与对应评审结论")
                        # Word 必须与回滚后的正文同源：此前只回滚 md，用户下载到的
                        # docx 仍是较差那一轮，等于「正文 80 分、附件 76 分」。
                        try:
                            from paper.generate_paper import md_to_docx as _rebuild
                            _rebuild(_md.read_text(encoding="utf-8"),
                                     out_dir / "06_paper" / "paper_full.docx", out_dir)
                            print("[EvalAgent] 已按最优轮正文重转 paper_full.docx")
                        except Exception as _de:  # noqa: BLE001
                            print(f"[EvalAgent] docx 重转失败（正文已回滚）：{_de}")
                    except Exception as e:  # noqa: BLE001
                        print(f"[EvalAgent] 回滚失败（保留最后一轮稿件）：{e}")
            # 论文文本诊断（给人读）：通读全文给「哪里有问题、怎么改」，不打分。
            # 与 eval_agent 分工：那边是分数与闸门，这边是指导教师式的文字诊断书。
            if os.environ.get("AUTO_PAPER_REVIEW", "1") != "0" and _md.exists():
                try:
                    pra = _load_module(ROOT / "agents" / "paper_review_agent.py", "pr_agent")
                    _pr = pra.run(str(_md), str(out_dir / "06_paper"), _verdict)
                    _npr = sum(len(c.get("problems") or [])
                               for c in (_pr.get("chapters") or [])) \
                        + len(_pr.get("problems") or [])
                    _nact = len(_pr.get("top_actions") or [])
                    result["paper_review"] = {
                        "verdict": _pr.get("verdict"), "n_problems": _npr,
                        "n_actions": _nact,
                        "report": str(out_dir / "06_paper" / "08_paper_review.md")}
                    print(f"[论文诊断] {_pr.get('verdict')} | {_npr} 处问题 | "
                          f"{_nact} 条优先动作 -> 06_paper/08_paper_review.md")
                    try:
                        from engine import feedback as FB
                        _nn = 0
                        # 规则兜底只数得出篇幅，入台账会变成「按该章补足内容」
                        # 这种空泛噪音，还会被 lessons() 当成反复出现的问题。
                        _actions = (_pr.get("top_actions") or [])[:6] \
                            if _pr.get("llm_available") else []
                        for _a in _actions:
                            FB.add(str(_a.get("action") or "")[:300], type="其他",
                                   severity="P1", stage="paper_review",
                                   run_dir=str(out_dir),
                                   topic=str(spec.get("title") or spec.get("topic") or ""),
                                   source="paper_review",
                                   evidence=str(_a.get("where") or ""),
                                   action=str(_a.get("why") or ""),
                                   provenance=("model" if _pr.get("llm_available") else "rule"),
                                   confidence="medium")
                            _nn += 1
                        if _nn:
                            print(f"[论文诊断] {_nn} 条改进动作已入反馈台账")
                    except Exception:
                        pass
                except Exception as _pe:  # noqa: BLE001
                    print(f"[论文诊断] 跳过：{_pe}")

        except Exception as e:  # noqa: BLE001
            result["errors"].append(f"generate_paper 异常: {e}")

        # 交付状态：模型没参与写作就明确标「不可交付」，防止半成品被当成品
        _deliv = True
        try:
            _dstat = mark_delivery(out_dir, _llm_written)
            _deliv = bool(_dstat.get("deliverable", True))
        except Exception as e:  # noqa: BLE001
            print(f"[交付状态] 判定跳过：{e}")
        try:
            trace.record(out_dir, "write+review", time.time() - _t_stage,
                         status=("error" if result["errors"] else "ok"),
                         detail=("deliverable" if _deliv else "degraded"))
        except Exception:
            pass

        # 6) 文献使用反馈：记录本次哪些论文真的被正文引用，供后续检索优先复用
        try:
            from engine import lit_feedback as LFB
            _mdp = out_dir / "06_paper" / "paper_full.md"
            _litp = out_dir / "01_literature.json"
            if _mdp.exists() and _litp.exists():
                _papers = json.loads(_litp.read_text(encoding="utf-8")).get("papers") or []
                _fb = LFB.record(str(spec.get("title") or spec.get("topic") or ""),
                                 _papers, _mdp.read_text(encoding="utf-8"))
                (out_dir / "05_lit_feedback.json").write_text(
                    json.dumps(_fb, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[Literature] 使用反馈：入选 {_fb['n_selected']} 篇，正文实际引用 {_fb['n_cited']} 篇")
                result["lit_feedback"] = {"n_selected": _fb["n_selected"],
                                          "n_cited": _fb["n_cited"]}
        except Exception as e:  # noqa: BLE001
            print(f"[Literature] 使用反馈记录失败（不影响主流程）：{e}")

    # 7) 闸门台账：把散落的检查点收成一张表。只记录与裁决，不改变各阶段原有行为。
    #    刷新两次：先拿裁决写进 trace 的 detail，再让 G11 把这一步也算进轨迹。
    try:
        from engine import gates as G
        _gr = G.refresh(out_dir, strict=True)
        result["gates"] = {
            "deliverable": _gr["decision"]["deliverable"],
            "blocking": _gr["blocking"],
            "unverified": _gr["unverified"],
            "report": str(out_dir / G.REPORT_FILE),
        }
        trace.record(out_dir, "gates", 0.0, status="ok",
                     detail=("deliverable" if _gr["decision"]["deliverable"] else "blocked"))
        G.refresh(out_dir, strict=True)
        # 交付状态与闸门裁决必须一致：mark_delivery 只看字数与模型参与，不看评审幻觉。
        G.sync_delivery(out_dir, _gr)
    except Exception as e:  # noqa: BLE001
        print(f"[闸门台账] 刷新失败（不影响主流程）：{e}")

    return result


def _paper_type_of_out(out_dir) -> str:
    """从产物目录的 task_spec.json 读取论文类型（读不到按方法对比型处理）。"""
    try:
        _p = Path(out_dir) / "task_spec.json"
        if _p.exists():
            return json.loads(_p.read_text(encoding="utf-8")).get("paper_type", "method_comparison")
    except Exception:
        pass
    return "method_comparison"


def _strict_gate_exit(res: dict, allow_degraded: bool, make_paper: bool):
    """严格模式的收尾裁决：返回需要提前退出的码；None 表示继续。

    两个入口（run / from_title）共用，避免「入口查了网关、中途写挂却照样返回 0」。
    这里只认明确的 fail；P0 的 unknown 是「本轮没跑到那一步」，只提示不误杀
    （--no-paper 时本来就不会有正文/评审）。
    """
    if allow_degraded or not make_paper:
        return None
    out = Path(res["out_dir"])
    st = out / "06_paper" / "DELIVERY_STATUS.json"
    if st.exists():
        try:
            _d = json.loads(st.read_text(encoding="utf-8"))
        except Exception:
            _d = {}
        if _d.get("deliverable") is False:
            print("\n[未产出可交付论文] " + "；".join(_d.get("reasons") or []))
            print(f"  半成品与中间产物保留在 {out}（已标【不可交付】），可直接续跑。")
            return 3
    g = res.get("gates") or {}
    blocking = g.get("blocking") or []
    unverified = g.get("unverified") or []
    if blocking:
        print("\n[未产出可交付论文] P0 闸门未通过：" + "、".join(blocking))
        print(f"  逐项证据见 {g.get('report') or (out / gates.REPORT_FILE)}")
        return 3
    if unverified:
        print("[提示] 以下 P0 项未验证（本轮没跑到对应步骤）：" + "、".join(unverified))
    return None


def _merge_spec_extra(spec: dict, path: str | None) -> dict:
    """把用户提供的 JSON 素材合并进 spec。

    约定：文件里若含 ``material`` 字段（dict），则视为「论文类型素材」，
    按 paper_type 声明的 spec_block（system_spec / case_spec / project_spec /
    proposal_spec）挂载；其余顶层键直接覆盖 spec。
    """
    if not path:
        return spec
    fp = Path(path)
    if not fp.exists():
        print(f"[warn] --spec-extra 文件不存在：{path}")
        return spec
    try:
        extra = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] --spec-extra 解析失败：{e}")
        return spec
    if not isinstance(extra, dict):
        return spec
    mat = extra.pop("material") if isinstance(extra.get("material"), dict) else None
    for k, v in extra.items():
        if k != "paper_type":
            spec[k] = v
    if mat:
        pt = spec.get("paper_type") or "method_comparison"
        try:
            from paper.paper_type_specs import TYPE_SPECS
            block_key = (TYPE_SPECS.get(pt) or {}).get("spec_block")
        except Exception:  # noqa: BLE001
            block_key = None
        if block_key:
            spec[block_key] = mat
            print(f"[spec-extra] 已挂载 {len(mat)} 个字段到 {block_key}（paper_type={pt}）")
        else:
            print(f"[warn] paper_type={pt} 不消费类型素材，已忽略 material")
    return spec


def cmd_run(args) -> int:
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    spec["paper_type"] = args.paper_type
    spec = _merge_spec_extra(spec, getattr(args, "spec_extra", ""))
    res = run_task(spec, args.out, make_paper=not args.no_paper, mode=args.mode)
    # 中途断网的情形：开跑时模型可用，跑到写作时挂了 → 产物已生成但不可交付。
    # 严格模式下这里必须失败退出，避免「悄悄给你一份半成品」。
    _exit = _strict_gate_exit(res, getattr(args, "allow_degraded", False),
                              not args.no_paper)
    if _exit is not None:
        return _exit

    a = res["analysis"] or {}
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
    _pt = _paper_type_of_out(res.get("out_dir"))
    if (_is_exp(_pt) and a.get("task_type") != "multi_dataset"
            and (not a.get("metrics") or a.get("best_method") is None)):
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
    elif getattr(args, "catalog_key", ""):
        try:
            from engine.topic_parser import DATASET_CATALOG, _data_binding as _bind
            _ds = next((x for x in DATASET_CATALOG
                        if x.get("key") == args.catalog_key), None)
        except Exception as e:  # noqa: BLE001
            print(f"[error] 无法读取数据集目录：{e}")
            return 1
        if _ds is None:
            print(f"[error] 数据集目录里没有 key={args.catalog_key}")
            return 1
        data_override = _bind(_ds)
        print(f"[catalog] 绑定本地数据集：{_ds['label']}（source={_ds['source']}）")

    # 2) Planner 出 spec（题目匹配到本地数据集目录时，走确定性快路径，可复现且
    #    能正确识别分类/回归/聚类/关联规则等任务类型，不依赖 LLM Planner 猜测）
    client, model, extra = planner.llm_client()
    if client is None:
        print("[warn] 未检测到 API Key（.env 中 OPENAI_API_KEY），使用模板 spec 降级运行。")
    spec = None
    _pt = args.paper_type
    if data_override is None:
        try:
            from engine.topic_parser import match_dataset as _match_ds, assemble_title_spec as _ats
            if _match_ds(args.topic) is not None:
                spec = _ats(args.topic)
                print("[plan·确定性] 题目匹配本地数据集目录，使用可复现 spec（无需 LLM Planner）")
        except Exception:
            spec = None
    if spec is None and data_override is not None:
        # 用户自带数据时，若题目点名关联规则等新类型，仍用确定性解析确定任务类型；
        # 图 / 时序等非表格源同样必须走确定性路径，否则 LLM Planner 会把它们
        # 当成普通 sklearn 数据集，绑定出来的 spec 根本加载不了。
        try:
            from engine.topic_parser import detect_task_type as _dtt, assemble_title_spec as _ats
            if (getattr(args, "catalog_key", "")
                    or _dtt(args.topic) == "association"
                    or data_override.get("source") in
                    ("graph", "timeseries", "association", "image")):
                spec = _ats(args.topic, data_override=data_override)
                print("[plan·确定性] 自带/目录数据 + 非表格任务，使用可复现 spec")
        except Exception:
            spec = None
    if spec is None:
        dataset_meta = _dataset_meta(data_override)
        pa = _load_module(ROOT / "agents" / "planning_agent.py", "planning_agent")
        spec = pa.run(args.topic, client, model, extra, data_override, dataset_meta=dataset_meta)
    if spec is not None:
        spec["paper_type"] = _pt
        spec = _merge_spec_extra(spec, getattr(args, "spec_extra", ""))
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
                out_dir=out_dir, top_n=_lit_top_n())
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
                    out_dir=out_dir, top_n=_lit_top_n())
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


def cmd_step(args) -> int:
    """只跑一个步骤（对应「模块可单独调用」）：plan / experiment / write / eval / polish。

    每一步只读写自己那份产物，不依赖前一步的进程状态：
      plan       → task_spec.json（默认用题目解析器，确定性；--planner 走 LLM 规划）
      experiment → 03_experiment.json（读 task_spec.json）
      write      → 06_paper/paper_full.{md,docx}（读 task_spec + 02/03/01）
      eval       → 06_paper/07_eval_verdict.json（读 paper_full.md）
      polish     → 06_paper/paper_full_polished.md（读 paper_full.md，纯规则）
    """
    name = args.name
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if name == "plan":
        if not args.topic:
            print("[step plan] 需要 --topic")
            return 1
        if args.planner:
            pa = _load_module(ROOT / "agents" / "planning_agent.py", "planning_agent")
            data = ({"source": "sklearn", "name": args.dataset} if args.dataset else None)
            spec = pa.run(args.topic, data_override=data, out_path=out / "task_spec.json")
            print("[step plan] 已用 LLM 规划器生成 spec")
        else:
            import engine.topic_parser as tp
            data = ({"source": "sklearn", "name": args.dataset} if args.dataset else None)
            spec = tp.assemble_title_spec(args.topic, data_override=data)
            (out / "task_spec.json").write_text(
                json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[step plan] 确定性解析：数据集={spec.get('_match_note')}")
        print(f"  产物：{out / 'task_spec.json'}")
        print("  下一步：python auto_research.py step --name experiment --out " + str(out))
        return 0

    if name == "experiment":
        spec_p = out / "task_spec.json"
        if not spec_p.exists():
            print(f"[step experiment] 缺少 {spec_p}，先跑 step --name plan")
            return 1
        spec = json.loads(spec_p.read_text(encoding="utf-8"))
        ea = _load_module(ROOT / "agents" / "experiment_agent.py", "experiment_agent")
        _t0 = time.time()
        res = ea.run(spec, out / "03_experiment.json")
        trace.record(out, "analysis", time.time() - _t0,
                     status=("error" if res.get("error") else "ok"),
                     detail=str(res.get("best_method") or ""))
        if res.get("error"):
            print("[step experiment] 失败：", res["error"])
            return 1
        print(f"[step experiment] 完成：best={res.get('best_method')} "
              f"value={res.get('best_value')} metrics={len(res.get('metrics') or [])}")
        print("  下一步：python auto_research.py step --name write --out " + str(out))
        return 0

    if name == "write":
        wa = _load_module(ROOT / "agents" / "writing_agent.py", "writing_agent")
        _t0 = time.time()
        r = wa.run(out, out / "task_spec.json", timeout=args.timeout)
        print(f"[step write] returncode={r['returncode']} paper={r['paper']}")
        _d = mark_delivery(out, r.get("llm_written"))
        trace.record(out, "write", time.time() - _t0,
                     status=("ok" if r["returncode"] == 0 else "error"),
                     detail=("deliverable" if _d.get("deliverable") else "degraded"))
        try:
            gates.refresh(out)
        except Exception:
            pass
        if r["returncode"] != 0:
            print(r["stderr"][-800:])
            return 1
        print("  下一步：step --name eval 或 step --name polish")
        return 0

    if name == "eval":
        md = out / "06_paper" / "paper_full.md"
        if not md.exists():
            print(f"[step eval] 缺少 {md}，先跑 step --name write")
            return 1
        ea = _load_module(ROOT / "agents" / "eval_agent.py", "eval_agent")
        # 与主流水线同口径：把 01_literature.json 的精选文献喂进去，
        # 否则单独跑 step eval 时文献维度会按「无文献」打分，与主流程的分数对不上。
        _cands = None
        _litp = out / "01_literature.json"
        if _litp.exists():
            try:
                _d = json.loads(_litp.read_text(encoding="utf-8"))
                _cands = _d if isinstance(_d, list) else (_d.get("papers") or [])
            except Exception:
                _cands = None
        v = ea.run(str(md), lit_candidates=_cands,
                   out_path=out / "06_paper" / "07_eval_verdict.json")
        _lj = v.get("llm_judge") or {}
        print(f"[step eval] 综合 {v.get('overall')}｜硬门 {v.get('gate')}｜"
              f"通过={v.get('passed')}｜LLM评审官="
              f"{'已介入' if _lj.get('available') else '未介入(' + str(_lj.get('reason') or '—') + ')'}")
        try:  # 自动采集：评审问题进反馈台账，供后续写作规避
            from engine import feedback as FB
            spec_p = out / "task_spec.json"
            topic = ""
            if spec_p.exists():
                topic = json.loads(spec_p.read_text(encoding="utf-8")).get("title", "")
            n = FB.record_eval_verdict(v, str(out), topic)
            if n:
                print(f"[step eval] 已记录 {n} 条反馈 → memory/feedback.jsonl")
        except Exception as e:  # noqa: BLE001
            print(f"[step eval] 反馈记录跳过：{e}")
        try:
            gates.refresh(out)
        except Exception:
            pass
        return 0

    if name == "review":
        md = out / "06_paper" / "paper_full.md"
        if not md.exists():
            print(f"[step review] 缺少 {md}，先跑 step --name write")
            return 1
        vp = out / "06_paper" / "07_eval_verdict.json"
        v = None
        if vp.exists():
            try:
                v = json.loads(vp.read_text(encoding="utf-8"))
            except Exception:
                v = None
        pra = _load_module(ROOT / "agents" / "paper_review_agent.py", "pr_agent")
        r = pra.run(str(md), str(out / "06_paper"), v)
        n = sum(len(c.get("problems") or []) for c in (r.get("chapters") or [])) \
            + len(r.get("problems") or [])
        print(f"[step review] {r.get('verdict')} | {n} 处问题 | "
              f"{len(r.get('top_actions') or [])} 条优先动作")
        print(f"  给人读的诊断书：{out / '06_paper' / '08_paper_review.md'}")
        try:
            from engine import feedback as FB
            cnt = 0
            _actions = (r.get("top_actions") or [])[:6] \
                if r.get("llm_available") else []
            for a in _actions:
                FB.add(str(a.get("action") or "")[:300], type="其他", severity="P1",
                       stage="paper_review", run_dir=str(out), source="paper_review",
                       evidence=str(a.get("where") or ""), action=str(a.get("why") or ""),
                       provenance=("model" if r.get("llm_available") else "rule"),
                       confidence="medium")
                cnt += 1
            if cnt:
                print(f"[step review] {cnt} 条改进动作已入反馈台账")
        except Exception as e:  # noqa: BLE001
            print(f"[step review] 反馈记录跳过：{e}")
        return 0

    if name == "polish":
        md = out / "06_paper" / "paper_full.md"
        if not md.exists():
            print(f"[step polish] 缺少 {md}，先跑 step --name write")
            return 1
        pol = _load_module(ROOT / "agents" / "polish_agent.py", "polish_agent")
        _t0 = time.time()
        _rep = pol.run(str(md))
        trace.record(out, "polish", time.time() - _t0,
                     detail=f"changes={_rep.get('total', 0)}")
        return 0

    print(f"[step] 未知步骤：{name}")
    return 1


def cmd_trace(args) -> int:
    """查看某次运行的执行轨迹：各阶段耗时、状态、最慢的一步。"""
    from engine import trace

    sm = trace.summary(args.out)
    if not sm["n_stages"]:
        print(f"{args.out} 下没有执行记录（00_trace.jsonl）。")
        return 1
    print(f"共 {sm['n_stages']} 个阶段，累计 {sm['total_s']}s"
          + (f"，最慢：{sm['slowest']['stage']}（{sm['slowest']['duration_s']}s）"
             if sm["slowest"] else ""))
    for r in sm["stages"]:
        flag = "" if r["status"] == "ok" else f" [{r['status']}]"
        print(f"  {r['ts'][11:]}  {r['stage']:14s} {r['duration_s']:8.2f}s  {r['detail']}{flag}")
    if sm["failed"]:
        print(f"失败阶段：{'、'.join(sm['failed'])}")
    return 0


def cmd_gates(args) -> int:
    """闸门台账：查看分级检查与交付裁决，或留痕人工放行。"""
    from engine import gates as G

    out = Path(args.out)
    if args.approve:
        if not args.who or not args.why:
            print("[闸门] 人工放行必须同时给 --who 和 --why（不放空口放行）")
            return 1
        r = G.approve(out, [x.strip() for x in args.approve.split(",") if x.strip()],
                      args.who, args.why)
        print("[闸门] 人工放行：" + json.dumps(r, ensure_ascii=False))
    rep = G.report(out, strict=not args.no_strict)
    # 交付状态必须与闸门裁决同源：step write 会先写一份「可交付」，
    # 之后 step eval 可能判 hard_fail。不回写就会出现两份文件互相矛盾。
    _ds = out / "06_paper" / "DELIVERY_STATUS.json"
    if _ds.exists():
        try:
            _d = json.loads(_ds.read_text(encoding="utf-8"))
            _d["deliverable"] = bool(rep["decision"]["deliverable"])
            _d["gate_decision"] = {
                "deliverable": rep["decision"]["deliverable"],
                "basis": rep["decision"].get("basis"),
                "report": "07_gate_report.json",
            }
            _ds.write_text(json.dumps(_d, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    print(json.dumps(rep, ensure_ascii=False, indent=2) if args.json else G.render(rep))
    if args.exit_code and not rep["decision"]["deliverable"]:
        return 3
    return 0


def cmd_backends(args) -> int:
    """领域后端：列出接线点、已装依赖、缺口优先级，并校验 ANALYSIS_JSON 契约。"""
    from engine import backend_contract as BC

    print("=== 四个接线点 ===")
    for k, v in BC.WIRING.items():
        print(f"  {k:5s} {v}")
    print("\n=== 已接线后端 ===")
    for s in BC.specs():
        print(f"  {s.name:12s} source={','.join(s.sources)}｜依赖={BC.check_deps(s.deps)}")
        print(f"      {s.note}")
    print("\n=== 缺口（按性价比） ===")
    for g in BC.gaps():
        print(f"  P{g['priority']} {g['name']}｜依赖就绪={g['deps_ready']}"
              f"｜需新评估分支={g['new_branch']}")
        print(f"      {g['how']}")
        print(f"      依赖状态 {g['dep_status']}；{g['extra']}")
    if args.probe:
        deps = sorted({d for s in BC.specs() for d in s.deps}
                      | {d for g in BC.GAPS for d in g["deps"]})
        print("\n=== 依赖真导入（子进程，含冷启动耗时） ===")
        for name, r in BC.probe(deps).items():
            if r.get("ok"):
                print(f"  {name:14s} ok   {r['s']:6.2f}s  version={r.get('version') or '?'}")
            else:
                print(f"  {name:14s} FAIL {r['s']:6.2f}s  {r.get('error')}")
    if args.check:
        ok, issues = BC.validate_file(args.check)
        print(f"\n=== 契约校验 {args.check}：{'通过' if ok else '未通过'} ===")
        for i in issues:
            print("  -", i)
        if not ok:
            return 1
    print("\n操作手册：docs/BACKENDS.md")
    return 0


def cmd_feedback(args) -> int:
    """反馈台账：查看统计/教训、手工补记、标记修复、导出回归用例。"""
    from engine import feedback as FB

    if args.stats:
        print(json.dumps(FB.stats(), ensure_ascii=False, indent=2))
        return 0
    if args.lessons:
        items = FB.lessons(min_count=args.min_count)
        print(json.dumps(items, ensure_ascii=False, indent=2) if items
              else "暂无反复出现的问题（同一类型出现 ≥2 次才会成为教训）")
        return 0
    if args.export_cases:
        print(json.dumps(FB.export_cases(), ensure_ascii=False, indent=2))
        return 0
    if args.fix:
        print("已标记" if FB.mark(args.fix, args.status) else "序号越界")
        return 0
    if args.add:
        rec = FB.add(args.add, type=args.type, severity=args.severity,
                     stage=args.stage, run_dir=args.out or "", topic=args.topic,
                     source="manual", provenance=args.provenance,
                     confidence=args.confidence)
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        return 0
    # 默认：最近 10 条
    rows = FB.load()[-10:]
    for r in rows:
        print(f"{r['ts']} [{r['type']}/{r['severity']}] {r['detail'][:70]}")
    if not rows:
        print("台账为空。用 --add 手工补记，或跑 step --name eval 自动采集。")
    return 0



def cmd_lit_pref(args) -> int:
    """文献选材偏好：人对「抓什么、选什么、别选什么」的表态，跨运行生效。"""
    from engine import selection_ledger as SL

    if args.selection:
        led = Path(args.selection) / SL.LEDGER_NAME
        if not led.exists():
            print(f"未找到选材台账：{led}")
            return 1
        d = json.loads(led.read_text(encoding="utf-8"))
        dec = d.get("decisions") or []
        print(f"题目：{d.get('topic')}")
        print(f"候选 {d.get('n_candidates')} 篇 → 入选 {d.get('n_selected')} 篇"
              f"（决策者 {d.get('selector')}）")
        by = {}
        for x in dec:
            by[x.get("by")] = by.get(x.get("by"), 0) + 1
        print("决策来源分布：" + "、".join(f"{k} {v} 条" for k, v in by.items()))
        rev = d.get("review") or {}
        if rev.get("missing_themes"):
            print("审评缺口：" + "；".join(map(str, rev["missing_themes"][:5])))
        if rev.get("weak_ids"):
            print("审评判定弱相关：" + "、".join(map(str, rev["weak_ids"][:8])))
        print("入选明细（前 12 条）：")
        for x in [y for y in dec if y.get("selected")][:12]:
            print(f"  [{x.get('by')}] {str(x.get('title'))[:70]} — {str(x.get('reason'))[:60]}")
        return 0

    acted = False
    if args.drop:
        r = SL.add_decision(args.drop, "drop", args.note, args.topic)
        print("已排除：" + json.dumps(r, ensure_ascii=False))
        acted = True
    if args.keep:
        r = SL.add_decision(args.keep, "keep", args.note, args.topic)
        print("已要求保留：" + json.dumps(r, ensure_ascii=False))
        acted = True
    if args.kind:
        r = SL.add_rule(args.kind, args.field, args.value, args.note,
                        args.scope, args.weight)
        print("已新增偏好规则：" + json.dumps(r, ensure_ascii=False))
        acted = True
    if args.review_mark:
        rd, iid, status = args.review_mark
        detail = ""
        rp = Path(rd) / "06_paper" / "08_paper_review.json"
        if rp.exists():
            try:
                rv = json.loads(rp.read_text(encoding="utf-8"))
                for a in rv.get("top_actions") or []:
                    if a.get("id") == iid:
                        detail = a.get("action") or ""
                        break
                if not detail:
                    for ch in rv.get("chapters") or []:
                        for pr in ch.get("problems") or []:
                            if pr.get("id") == iid:
                                detail = f"{pr.get('why') or ''} → {pr.get('how') or ''}"
                                break
            except Exception:
                pass
        r = SL.add_review_note(rd, iid, status, args.note, detail)
        print("已记下评审批注：" + json.dumps(r, ensure_ascii=False))
        acted = True

    if acted or args.list:
        d = SL.load_prefs()
        print("\n=== 当前偏好（memory/lit_preferences.json）===")
        for r in d.get("rules", []):
            print(f"  {r['id']} [{r['kind']}/{r['field']}] {r['value']}"
                  f"  scope={r['scope']}  {r.get('note') or ''}")
        for x in d.get("paper_decisions", []):
            print(f"  单篇 {x['action']}: {str(x.get('raw'))[:70]}  {x.get('note') or ''}")
        for x in d.get("review_notes", [])[-5:]:
            print(f"  评审批注 {x['item_id']}/{x['status']}: {str(x.get('detail'))[:60]}")
        pp = SL.preference_prompt(args.topic)
        if pp:
            print("\n" + pp)
        return 0
    print("用法：lit-pref --list / --selection <目录> / --kind avoid --field keyword "
          "--value preprint / --drop <题名或DOI> / --keep <题名或DOI> / "
          "--review-mark <目录> <意见编号> accepted|rejected|done")
    return 0


def cmd_doctor(args) -> int:
    """一条命令自检：模型网关 + 文献数据源是否可用。明天想知道「好了没」就跑它。"""
    import time
    from engine import llm_client as LLM

    try:  # 先载入 .env，否则这里读到的是空值
        LLM._load_dotenv()
    except Exception:
        pass
    base = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("OPENAI_MODEL", "")
    print("=== 模型网关 ===")
    print(f"  地址：{base or '（未配置）'}")
    print(f"  模型：{model or '（未配置）'}")
    try:
        chs = LLM.channels()
        cur = LLM.current_channel()
        print(f"  通道：已配置 {len(chs)} 条｜当前第 {cur + 1} 条｜"
              f"连续失败 {LLM._FAILOVER_AFTER} 次自动切备用｜调用最小间隔 {LLM._RL_MIN_INTERVAL}s")
        for i, ch in enumerate(chs, 1):
            print(f"    {i}) {ch['label']}：{ch['base']} / {ch['model']}"
                  + ("  ← 当前" if i - 1 == cur else ""))
        if len(chs) < 2:
            print("    备用通道未配置：.env 里加 OPENAI_BASE_URL_2 / OPENAI_API_KEY_2 / "
                  "OPENAI_MODEL_2 即可，挂了会自动切")
    except Exception as e:  # noqa: BLE001
        print(f"  通道信息读取失败：{e}")
    t0 = time.time()
    ok = False
    try:
        ok = LLM.available()
    except Exception as e:  # noqa: BLE001
        print(f"  探测异常：{e}")
    print(f"  结果：{'可用' if ok else '不可用（已确认 6 秒内快速失败）'}｜耗时 {time.time() - t0:.1f}s")
    if not ok:
        print("  → 论文写作与 LM 读文献会走确定性兜底；不影响实验/排版链路")

    print("\n=== 文献数据源 ===")
    _mailto = os.environ.get("LIT_CONTACT_EMAIL", "").strip()
    print(f"  礼貌池邮箱：{_mailto or '未配置（OpenAlex 不带 mailto 时更容易 429）'}")
    probes = []
    try:
        import literature as lit_mod
        from agents import literature_agent as LA
        probes += [
            ("crossref", lambda: LA.fetch_crossref("machine learning", max_results=2)),
            ("arxiv", lambda: lit_mod.fetch_arxiv("machine learning", max_results=2)),
            ("s2", lambda: LA.fetch_semantic_scholar("machine learning", max_results=2)),
            ("openalex", lambda: LA.fetch_openalex("machine learning", max_results=2)),
            ("europepmc", lambda: LA.fetch_europepmc("machine learning", max_results=2)),
            ("local", lambda: LA.fetch_imported("machine learning", max_results=2)),
        ]
    except Exception as e:  # noqa: BLE001
        print(f"  文献模块加载失败：{e}")

    usable = []
    for name, fn in probes:
        t0 = time.time()
        try:
            got = fn()
            if name == "local":
                print(f"  {name:9s} 本地题录 {len(got)} 条"
                      + ("（corpus/imports/ 为空，可从知网/Scholar 导出 .bib 放入）"
                         if not got else "（不联网，永不限流）"))
            else:
                print(f"  {name:9s} 可用｜{len(got)} 条｜{time.time() - t0:.1f}s")
            if got or name == "local":
                usable.append(name)
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace("\n", " ")[:80]
            print(f"  {name:9s} 不可用｜{msg}｜{time.time() - t0:.1f}s")

    print("\n=== 结论 ===")
    if ok:
        print("  模型可用：LM 读文献、LLM 写作、评审闭环都能跑。")
    else:
        print("  模型不可用：先跑确定性链路（--no-paper 或直接跑，会标 confidence=keyword_only）。")
    if usable:
        print(f"  文献源可用：{'、'.join(usable)}")
    else:
        print("  文献源全部不可用：检查网络，或过几分钟再看（429 需要冷却）。")
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
    spec["paper_type"] = args.paper_type
    spec = _merge_spec_extra(spec, getattr(args, "spec_extra", ""))
    if _is_exp(args.paper_type) and spec.get("data") is None:
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

    # 默认严格：只要高质量稿，不接受降级稿。需要出半成品自测时显式加 --allow-degraded。
    if not getattr(args, "allow_degraded", False):
        _wait = int(getattr(args, "wait_llm", 0) or 0)
        if not wait_for_llm(_wait):
            _out = Path(args.out)
            print("\n[未产出论文] 模型不可用，且未允许降级。")
            _have = _existing_artifacts(_out) if _out.exists() else []
            if _have:
                print(f"  已保留的中间产物：{'、'.join(_have)}（重跑不会重做这些）")
            print("  可选做法：")
            print("    1) 稍后重试（会自动复用已有产物）：加 --wait-llm 600，或过一会儿再跑同一条命令")
            print("    2) 只想先看实验/数据：加 --no-paper")
            print("    3) 确需半成品自测：加 --allow-degraded（产物会标【不可交付】）")
            return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_task(spec, out_dir, make_paper=not args.no_paper, mode=args.mode,
                   force=getattr(args, "force", False))
    a = res["analysis"] or {}
    summary = {
        "out_dir": res["out_dir"],
        "paper": res["paper"],
        "best_method": a.get("best_method"),
        "best_value": a.get("best_value"),
        "plots": [p.get("file") for p in a.get("plots", [])],
        "errors": res["errors"],
        "gates": res.get("gates"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if a.get("error"):
        print("分析层报错：", a["error"])
        return 1
    _pt = _paper_type_of_out(res.get("out_dir"))
    if (_is_exp(_pt) and a.get("task_type") != "multi_dataset"
            and (not a.get("metrics") or a.get("best_method") is None)):
        print("分析未产生有效结果。")
        return 1
    # 入口只查了网关；中途写挂/评审发现幻觉，这里必须同样失败退出。
    _exit = _strict_gate_exit(res, getattr(args, "allow_degraded", False),
                              not args.no_paper)
    if _exit is not None:
        return _exit
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
    rp.add_argument("--paper-type", default="method_comparison",
                    choices=PAPER_TYPE_CHOICES,
                    help="论文类型：method_comparison（默认，方法对比型实证）/ survey（文献综述型）/ "
                         "system_design（系统·软件设计型）/ case_study（案例研究型）/ "
                         "technical_report（工程实践报告型）/ research_proposal（研究方案·开题型）")
    rp.add_argument("--spec-extra", default="",
                    help="论文类型素材 JSON（含 material 字段时按类型挂到对应 spec_block）")
    rp.set_defaults(func=cmd_run)

    pp = sub.add_parser("plan", help="给一句研究主题，自动规划 spec + 拉文献 + 出论文")
    pp.add_argument("--topic", required=True, help="自然语言研究主题")
    pp.add_argument("--out", default="out_plan", help="产物输出目录")
    pp.add_argument("--dataset", default="", help="指定内置 sklearn 数据集名（覆盖 Planner 选择）")
    pp.add_argument("--catalog-key", default="",
                    help="按 DATASET_CATALOG 的 key 绑定本地数据集（图/时序/关联规则等非 sklearn 源用这个）")
    pp.add_argument("--data", default="", help="自带 CSV 路径（与 --target 配合）")
    pp.add_argument("--url", default="", help="公开数据集网址（CSV/XLSX，与 --target 配合）")
    pp.add_argument("--target", default="", help="CSV 中目标列名或列索引")
    pp.add_argument("--sep", default="", help="CSV 分隔符（如分号 ';'，默认逗号）")
    pp.add_argument("--na_values", default="", help="缺失值标记（如 '?'，多个用逗号分隔）")
    pp.add_argument("--no-literature", action="store_true", help="跳过 arXiv 文献检索")
    pp.add_argument("--no-paper", action="store_true", help="只跑分析，不出论文")
    pp.add_argument("--mode", default="general",
                    help="机构模式：general（默认，通用学术标准）或 sztu（深圳技术大学预设）")
    pp.add_argument("--paper-type", default="method_comparison",
                    choices=PAPER_TYPE_CHOICES,
                    help="论文类型：method_comparison（默认）/ survey（文献综述型）/ system_design（系统·软件设计型）")
    pp.add_argument("--spec-extra", default="",
                    help="论文类型素材 JSON（含 material 字段时按类型挂到对应 spec_block）")
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
    fp.add_argument("--allow-degraded", action="store_true",
                    help="允许在模型不可用时产出半成品（会标【不可交付】，仅用于自测）")
    fp.add_argument("--wait-llm", type=int, default=0,
                    help="模型不可用时最多等待多少秒（每 60s 重试一次），默认不等待")
    fp.add_argument("--require-llm", action="store_true",
                    help="（兼容旧参数）严格模式；现在默认即严格模式")
    fp.add_argument("--dry-run", action="store_true",
                    help="只预演解析结果（数据集/方法/检索式），不跑实验不出论文")
    fp.add_argument("--mode", default="general",
                    help="机构模式：general（默认，通用学术标准）或 sztu（深圳技术大学预设）")
    fp.add_argument("--paper-type", default="method_comparison",
                    choices=PAPER_TYPE_CHOICES,
                    help="论文类型：method_comparison（默认）/ survey（文献综述型）/ system_design（系统·软件设计型）")
    fp.add_argument("--spec-extra", default="",
                    help="论文类型素材 JSON（含 material 字段时按类型挂到对应 spec_block）")
    fp.set_defaults(func=cmd_from_title)

    sp = sub.add_parser("step", help="只跑某一步（plan/experiment/write/eval/review/polish），便于单模块调用")
    sp.add_argument("--name", required=True,
                    choices=["plan", "experiment", "write", "eval", "review", "polish"])
    sp.add_argument("--out", required=True, help="产物目录")
    sp.add_argument("--topic", default="", help="plan 步骤需要：题目/主题")
    sp.add_argument("--dataset", default="", help="plan 步骤可选：sklearn 数据集名")
    sp.add_argument("--planner", action="store_true", help="plan 用 LLM 规划器（默认确定性解析）")
    sp.add_argument("--timeout", type=int, default=1800, help="write 步骤子进程超时")
    sp.set_defaults(func=cmd_step)

    tp = sub.add_parser("trace", help="查看某次运行的执行轨迹（各阶段耗时/状态）")
    tp.add_argument("--out", required=True, help="产物目录")
    tp.set_defaults(func=cmd_trace)

    gp = sub.add_parser("gates", help="闸门台账：P0/P1/P2 分级检查、交付裁决、人工放行留痕")
    gp.add_argument("--out", required=True, help="产物目录")
    gp.add_argument("--json", action="store_true", help="输出完整 JSON 报告")
    gp.add_argument("--no-strict", action="store_true", help="仅查看：P0 未验证不阻塞交付")
    gp.add_argument("--exit-code", action="store_true", help="不可交付时返回退出码 3（供脚本/CI 用）")
    gp.add_argument("--approve", default="", help="人工放行的闸门 id，逗号分隔（如 G03,G05）")
    gp.add_argument("--who", default="", help="放行人")
    gp.add_argument("--why", default="", help="放行理由")
    gp.set_defaults(func=cmd_gates)

    fbp = sub.add_parser("feedback", help="反馈台账：统计/教训/补记/标记修复/导出回归用例")
    fbp.add_argument("--stats", action="store_true", help="按类型统计问题分布")
    fbp.add_argument("--lessons", action="store_true", help="输出写作前应规避的教训")
    fbp.add_argument("--export-cases", action="store_true", help="导出已修复问题的回归用例")
    fbp.add_argument("--add", default="", help="手工补记一条问题描述")
    fbp.add_argument("--type", default="其他", help="问题类型（数据错配/方法缺失/结构缺陷/内容矛盾/文献不足/写作风格/可复现性）")
    fbp.add_argument("--severity", default="P2", choices=["P0", "P1", "P2"])
    fbp.add_argument("--stage", default="manual")
    fbp.add_argument("--topic", default="")
    fbp.add_argument("--out", default="", help="关联的产物目录")
    fbp.add_argument("--fix", type=int, default=0, help="标记第 N 条的状态")
    fbp.add_argument("--status", default="fixed", choices=["fixed", "doing", "wontfix", "open"])
    fbp.add_argument("--min-count", type=int, default=2)
    fbp.add_argument("--provenance", default="human", choices=["human", "model", "rule"],
                     help="这条反馈的来源：人/模型/规则")
    fbp.add_argument("--confidence", default="medium", choices=["high", "medium", "low"],
                     help="这条反馈的把握程度")
    fbp.set_defaults(func=cmd_feedback)

    lpp = sub.add_parser("lit-pref", help="文献选材偏好：记住人抓什么/选什么/别选什么，下轮生效")
    lpp.add_argument("--list", action="store_true", help="列出全部偏好与单篇表态")
    lpp.add_argument("--selection", default="", help="查看某次运行的选材台账目录")
    lpp.add_argument("--drop", default="", help="排除某篇（题名/DOI/id），下轮不再入选")
    lpp.add_argument("--keep", default="", help="要求保留某篇，优先于排除规则")
    lpp.add_argument("--kind", default="", choices=["avoid", "require", "boost"])
    lpp.add_argument("--field", default="keyword",
                     choices=["id", "title", "venue", "year", "keyword", "topic"])
    lpp.add_argument("--value", default="", help="匹配值；year 支持 >=2022 / <=2020")
    lpp.add_argument("--scope", default="global", help="global 或题目关键词")
    lpp.add_argument("--weight", type=float, default=0.15, help="require/boost 的加分")
    lpp.add_argument("--note", default="", help="这条偏好/批注的理由")
    lpp.add_argument("--topic", default="", help="关联题目")
    lpp.add_argument("--review-mark", nargs=3, default=[],
                     metavar=("RUN_DIR", "ITEM_ID", "STATUS"),
                     help="对诊断报告某条意见表态：accepted/rejected/done")
    lpp.set_defaults(func=cmd_lit_pref)

    dcp = sub.add_parser("doctor", help="自检：模型网关与文献数据源是否可用")
    dcp.set_defaults(func=cmd_doctor)

    bp = sub.add_parser("backends", help="领域后端：接线点、依赖状态、缺口优先级与产物契约自检")
    bp.add_argument("--check", default="", help="校验某个 03_experiment.json 是否满足事实契约")
    bp.add_argument("--probe", action="store_true",
                    help="真导入一遍依赖并计时（find_spec 只查得到文件，查不到能不能用）")
    bp.set_defaults(func=cmd_backends)

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
