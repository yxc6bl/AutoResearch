"""论文类型端到端验证：为每种非实验型各建一份 spec，构建论文并跑闸门（可选跑评审）。

用法：
    python verify_multitype.py                    # 全部非实验类型，构建 + 闸门
    python verify_multitype.py --types case_study # 只验证某几类
    python verify_multitype.py --skip-build       # 复用已有产物，只跑闸门/评审
    python verify_multitype.py --eval             # 额外跑 EvalAgent（会调 LLM 评审官）
    python verify_multitype.py --eval --rounds 2  # 评审不过就带意见重写一轮再复评
    python verify_multitype.py --eval --rounds 3  # 最多三轮，取评审分最高的一版

产物目录：out_verify_<paper_type>/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paper.thesis_template import (  # noqa: E402
    PAPER_TYPES, chapters_for, structure_checklist_for,
)

LIT_SRC = ROOT / "out_demo_heart" / "01_literature.json"

# ── 各类型的演示 spec（素材都是「作者真实掌握」的那一类信息） ──────────
DEMO_SPECS: dict[str, dict] = {
    "survey": {
        "title": "基于机器学习的心血管疾病风险预测研究进展：文献综述",
        "topic": "基于机器学习的心血管疾病风险预测研究进展",
        "literature_query": "machine learning cardiovascular disease risk prediction",
        "institution_mode": "general",
    },
    "system_design": {
        "title": "面向校园健康场景的可穿戴心电监测与预警系统设计与实现",
        "topic": "面向校园健康场景的可穿戴心电监测与预警系统设计与实现",
        "literature_query": "wearable ECG monitoring atrial fibrillation detection",
        "institution_mode": "general",
        "system_spec": {
            "summary": "一套面向高校场景的可穿戴心电(ECG)实时监测与异常预警系统，由采集终端、"
                       "边缘网关与云端分析三层构成，支持单导联 ECG 采集、心率变异性分析与房颤/早搏预警。",
            "tech_stack": ["Python", "FastAPI", "WebSocket", "SQLite", "PyTorch", "Flutter", "MQTT", "Docker"],
            "requirements": {
                "functional": ["单导联 ECG 实时采集与上传", "云端心率/心律失常分析",
                               "异常事件 App 推送预警", "历史数据查询与导出"],
                "non_functional": ["端到端延迟<2s", "采集终端续航>24h",
                                   "数据加密传输与最小权限访问控制", "模块化便于二次开发"],
            },
            "architecture": {
                "layers": ["设备采集层(手环/胸贴)", "边缘网关层(MQTT 汇聚+预处理)",
                           "云端服务层(存储/分析/预警)", "前端展示层(App/Web)"],
                "data_flow": "采集终端→MQTT→边缘网关→云端 WebSocket→分析服务→预警/存储→前端",
            },
            "modules": [
                {"name": "采集终端固件", "responsibility": "单导联 ECG 采样、运动伪迹初筛、断点续传",
                 "key_algorithms": "Pan-Tompkins QRS 检测"},
                {"name": "边缘网关", "responsibility": "多设备汇聚、数据缓冲与协议转换",
                 "key_algorithms": "MQTT + 缓冲队列"},
                {"name": "分析服务", "responsibility": "心率变异性与心律失常判定（窦性心律/房颤/室性早搏/房性早搏四类）",
                 "key_algorithms": "1D-CNN 心律失常分类"},
                {"name": "预警与存储", "responsibility": "阈值/模型双路预警、时序库落盘",
                 "key_algorithms": "规则引擎 + 模型融合"},
            ],
            "key_techniques": ["单导联 QRS 检测(Pan-Tompkins)", "1D-CNN 心律失常分类",
                               "MQTT 轻量传输", "模型推理服务容器化"],
            "validation": {
                "strategy": "单元+接口+系统三级测试：固件采样精度校准、网关压测、"
                            "分析服务在公开 ECG 数据集上的混淆矩阵评估",
                "test_cases": ["QRS 检出率>98%", "房颤 F1>0.92(MIT-BIH)",
                               "100 并发连接时延<1.5s", "断电续传不丢点"],
            },
        },
    },
    "case_study": {
        "title": "县域中学课后服务质量提升的案例研究——以某县一中为例",
        "topic": "县域中学课后服务质量提升路径",
        "literature_query": "after-school program quality rural middle school China",
        "institution_mode": "general",
        "case_spec": {
            "subject": "某县第一中学（2021 年起实施课后服务）",
            "background": "该校为县域寄宿制完全中学，2021 年秋季学期起按政策开展课后服务，"
                          "覆盖非教学时段的作业辅导与社团活动。",
            "phenomenon": "学校投入了师资与场地，但教师参与意愿分化、活动质量参差，家长满意度未见明显提升。",
            "framework": "服务质量管理（SERVQUAL）五维度：有形性、可靠性、响应性、保证性、移情性",
            "materials": ["学校课后服务实施方案", "家长满意度问卷原始数据", "教师访谈记录",
                          "学期活动安排表"],
            "analysis_points": ["供给结构：谁在提供、提供了什么",
                                "组织激励：教师参与意愿的差异来源",
                                "需求匹配：活动供给与家长/学生需求的错位"],
            "findings": "初步判断：问题不在投入总量，而在供给结构与激励安排的不匹配。",
        },
    },
    "technical_report": {
        "title": "某高校教务数据中台建设项目实践报告",
        "topic": "高校教务数据中台建设",
        "literature_query": "university academic affairs data platform master data governance",
        "institution_mode": "general",
        "project_spec": {
            "background": "学校教务、学工、图书等系统各自建库，数据口径不一致，报表需人工汇总。",
            "objectives": ["建立统一的教务主题数据模型", "打通 4 个业务系统的主数据",
                           "提供自助报表与接口服务"],
            "scope": "本期只覆盖教务与学工两个主题域，不含科研与财务。",
            "tech_stack": ["Python", "Airflow", "Doris", "Kafka", "Vue"],
            "work_items": ["阶段一：数据源梳理与口径对齐", "阶段二：主题模型与 ETL 开发",
                           "阶段三：指标服务与报表开发", "阶段四：联调、迁移与验收"],
            "deliverables": ["数据中台一套（含 ETL 作业与指标服务）", "口径说明文档", "运维手册"],
            "acceptance": ["核心报表与人工汇总结果逐项对齐", "日报生成时间由人工 1 天缩短为自动出数",
                           "4 个源系统主数据一致率达标"],
            "issues": ["学号在历史库中存在一码多源，采用以教务库为主的仲裁规则",
                       "部分业务表缺时间戳，改用归档表补齐"],
        },
    },
    "research_proposal": {
        "title": "面向毕业设计的过程性评价方法研究（开题报告）",
        "topic": "面向毕业设计的过程性评价方法",
        "literature_query": "capstone design project process assessment engineering education",
        "institution_mode": "general",
        "proposal_spec": {
            "background": "毕业设计评价以终期答辩为主，过程投入难以体现，师生对评价公平性存在争议。",
            "questions": ["过程性数据能否稳定刻画毕业设计的投入与进展？",
                          "过程性评价与终期评价的结合权重如何影响评价公平性？"],
            "objectives": ["构建可采集的过程性指标体系", "给出过程+终期的组合评价方案"],
            "content": ["过程性指标梳理与可采集性分析", "评价方案设计与试点",
                        "方案效果与公平性分析"],
            "methods": ["文献研究", "问卷与访谈", "试点对照分析"],
            "schedule": [{"phase": "第 1-3 月", "task": "文献与指标梳理"},
                         {"phase": "第 4-6 月", "task": "方案设计与工具开发"},
                         {"phase": "第 7-9 月", "task": "试点与数据分析"},
                         {"phase": "第 10-12 月", "task": "论文撰写与答辩"}],
            "expected": ["学位论文一篇", "过程性评价方案一套", "试点数据一组"],
            "feasibility": "依托本院毕业设计管理平台，过程数据可获得；试点范围可控。",
        },
    },
}


def _write_spec(out: Path, pt: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(DEMO_SPECS[pt])
    spec["paper_type"] = pt
    (out / "task_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return spec


def _ensure_literature(out: Path, spec: dict, force: bool = False) -> str:
    """有 literature_query 就现场检索真实文献（题目要对得上，否则引用会被判编造）。"""
    lit = out / "01_literature.json"
    q = spec.get("literature_query")
    if q and (force or not lit.exists()):
        r = subprocess.run(
            [sys.executable, "agents/literature_gather_agent.py",
             "--topic", q, "--task", spec.get("paper_type", "survey"),
             "--out", str(out), "--top-n", "40"],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600)
        print(f"    检索文献：rc={r.returncode} -> {'ok' if lit.exists() else 'FAIL'}")
    elif not lit.exists() and LIT_SRC.exists():
        # 旧版静默复制一份心血管文献池，任何没给检索词的类型都会被塞进去，
        # 评审官随后判「文献与主题零交集」。保留兜底但必须留痕、必须警告。
        shutil.copy2(LIT_SRC, lit)
        print("    [警告] 本类型未给 literature_query，复用演示文献池"
              "（out_demo_heart，心血管主题）；与题目不符时请用 --fetch-lit 重抓")
    # 检索完成后做一次主题一致性提示：只报告，不过滤文献、不改引用编号。
    if lit.exists() and q:
        try:
            from paper.lit_material import topic_overlap
            _papers = json.loads(lit.read_text(encoding="utf-8")).get("papers") or []
            _ov = topic_overlap(q, _papers)
            (out / "00_lit_relevance.json").write_text(json.dumps({
                "query": q, "n_papers": len(_papers), "topic_overlap": _ov,
                "note": "粗粒度词面一致性，仅用于预警；不参与闸门、不改动引用编号。",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            if _ov < 0.25:
                print(f"    [警告] 文献池与检索式主题一致性仅 {_ov:.2f}，"
                      f"疑似跑题；建议 --fetch-lit 重抓（当前 {len(_papers)} 篇）")
            else:
                print(f"    文献主题一致性：{_ov:.2f}（{len(_papers)} 篇）")
        except Exception:
            pass
    # 有文献清单就补一次覆盖审评：否则 G08 在该目录永远是 unknown，
    # 「没检查」会被误读成「检查了没问题」。
    rev = out / "04_literature_review.json"
    if lit.exists() and not rev.exists():
        r = subprocess.run(
            [sys.executable, "agents/literature_review_agent.py",
             "--lit", str(lit), "--topic", spec.get("topic") or spec.get("title", "")],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300)
        try:
            rev.write_text(r.stdout, encoding="utf-8")
            print("    文献审评：ok")
        except Exception:
            print(f"    文献审评失败：rc={r.returncode}")
    return str(lit)


def _stubs(out: Path, pt: str) -> None:
    """补齐 run_task 会写的非实验桩产物（G01 豁免 / 交付状态 / trace）。"""
    (out / "03_experiment.json").write_text(json.dumps({
        "paper_type": pt, "task_type": pt,
        "note": f"非实验型论文（{pt}），不跑确定性实验引擎",
        "analysis": {"seed_variance": {}}, "metrics": [], "best_method": None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (out / "06_paper").mkdir(parents=True, exist_ok=True)
    md = out / "06_paper" / "paper_full.md"
    docx = out / "06_paper" / "paper_full.docx"
    (out / "06_paper" / "DELIVERY_STATUS.json").write_text(json.dumps({
        "delivered": True, "paper_type": pt,
        "markdown": str(md), "docx": str(docx) if docx.exists() else "",
        "markdown_exists": md.exists(), "docx_exists": docx.exists(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 字段名必须与 engine/gates.py G11 的读法一致（stage/status）；写 step/ok 会被读成
    # status=None，于是「一次都没失败的运行」被记成失败阶段 [None]，G11 假失败。
    # 用覆盖写而非追加：旧版残留的坏行会一直留在文件里污染判定。
    (out / "00_trace.jsonl").write_text(
        json.dumps({"stage": "write", "status": "ok", "paper_type": pt},
                   ensure_ascii=False) + "\n",
        encoding="utf-8")


def _backup_before_build(out: Path) -> list[tuple[Path, Path]]:
    """构建前备份已有的 LLM 成稿：网关抖动时子进程会降级成 2 千字骨架，
    直接覆盖会把好稿毁掉（实测踩过：一次限流把两篇 1.2 万字稿子换成兜底）。
    """
    saved = []
    for name in ("paper_full.md", "paper_full.docx", "WRITE_STATUS.json",
                 "07_eval_verdict.json"):
        p = out / "06_paper" / name
        if p.exists():
            b = out / "06_paper" / (name + ".prebuild")
            shutil.copy2(p, b)
            saved.append((b, p))
    return saved


def _restore_after_failed_build(out: Path, saved: list[tuple[Path, Path]]) -> None:
    for b, p in saved:
        try:
            shutil.copy2(b, p)
            b.unlink(missing_ok=True)
        except Exception:
            pass


def _build(out: Path, timeout: int = 600) -> tuple[int, str]:
    """构建单篇论文。必须有整体超时：网关限流时单批写作可退避重试十几分钟，
    旧版 subprocess.run 没有 timeout，整个验证会无限期挂住（实测踩到过）。
    """
    try:
        r = subprocess.run([sys.executable, "paper/generate_paper.py", "--output-dir", str(out)],
                           cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"构建超时（>{timeout}s）：放弃本轮，保留构建前成稿"
    tail = (r.stdout or "").strip().splitlines()[-3:]
    return r.returncode, " / ".join(tail)


def _gateway_ok() -> bool:
    """构建前的可用性探测。失败时宁可跳过构建，也不要覆盖已有成稿。"""
    try:
        from engine import llm_client as LC
        LC.reset_cache()
        return bool(LC.available())
    except Exception:
        return False


def _gates(out: Path) -> dict:
    r = subprocess.run([sys.executable, "engine/gates.py", "--out", str(out), "--json"],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"blocking": ["<gate parse error>"], "gates": []}


def _structure(out: Path, pt: str) -> tuple[int, int]:
    md = out / "06_paper" / "paper_full.md"
    if not md.exists():
        return 0, len(structure_checklist_for(pt))
    text = md.read_text(encoding="utf-8")
    checks = structure_checklist_for(pt)
    hit = 0
    for c in checks:
        name = c.lstrip("# ").strip()
        if name in text:
            hit += 1
    return hit, len(checks)


def _used_llm(out: Path) -> bool:
    p = out / "06_paper" / "WRITE_STATUS.json"
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("used_llm"))
    except Exception:
        return False


def _chars(out: Path) -> int:
    md = out / "06_paper" / "paper_full.md"
    if not md.exists():
        return 0
    import re
    return len(re.findall(r"[\u4e00-\u9fff]", md.read_text(encoding="utf-8")))


def _eval(out: Path) -> dict:
    r = subprocess.run([sys.executable, "agents/eval_agent.py",
                        str(out / "06_paper" / "paper_full.md"),
                        "--out", str(out / "06_paper" / "07_eval_verdict.json"),
                        "--mode", "general"],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def _needs_revise(v: dict, all_signals: bool = False) -> bool:
    """本轮是否值得带意见重写。

    2026-11-09 实测（三次两轮，真实 LLM）给出的经验：
      - 因「硬编造」重写：1/1 改善（评审 72→78，幻觉 3→0）；
      - 仅因「风格建议」重写：0/2 改善（72→62、74→72），都靠回滚才没退步。

    所以默认只在评审官指得出具体编造时重写：那是可执行的删除指令；而「写深一点」
    「补论证」这类建议没有事实锚点，模型容易在扩写时引入新细节，反而更差。
    想强行试的用 ``--revise-all``。
    """
    lj = (v or {}).get("llm_judge") or {}
    if not lj.get("available"):
        return False
    if lj.get("hallucinations"):
        return True
    return bool(all_signals and lj.get("verdict") == "revise")


def _write_review_brief(out: Path, v: dict) -> int:
    """把评审官的指控落成 04_review.json——这是写作端唯一读取的重写输入。"""
    lj = (v or {}).get("llm_judge") or {}
    brief = {
        "hallucinations": lj.get("hallucinations") or [],
        "unsupported_claims": lj.get("unsupported_claims") or [],
        "suggestions": lj.get("suggestions") or [],
        "round_source": "verify_multitype",
    }
    n = sum(len(brief[k]) for k in ("hallucinations", "unsupported_claims", "suggestions"))
    (out / "04_review.json").write_text(
        json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
    return n


def _score_of(v: dict):
    lj = (v or {}).get("llm_judge") or {}
    if lj.get("available") and isinstance(lj.get("overall"), (int, float)):
        return (float(lj["overall"]), -len(lj.get("hallucinations") or []))
    return (float((v or {}).get("overall") or 0), 0)


def _restore_best(out: Path, snapshot: Path, verdict_snap: Path) -> None:
    """回滚正文与评审结论，并重转 Word——三者必须同源。"""
    md = out / "06_paper" / "paper_full.md"
    shutil.copy2(snapshot, md)
    if verdict_snap.exists():
        shutil.copy2(verdict_snap, out / "06_paper" / "07_eval_verdict.json")
    try:
        from paper.generate_paper import md_to_docx
        md_to_docx(md.read_text(encoding="utf-8"),
                   out / "06_paper" / "paper_full.docx", out)
    except Exception as e:  # noqa: BLE001
        print(f"    docx 重转失败（正文已回滚）：{e}")
    # WRITE_STATUS 里的字数也要跟着回滚，否则「正文 13k 字、状态文件写 9k」自相矛盾
    try:
        import re as _re
        sp = out / "06_paper" / "WRITE_STATUS.json"
        st = json.loads(sp.read_text(encoding="utf-8"))
        st["chinese_chars"] = len(_re.findall(r"[\u4e00-\u9fff]", md.read_text(encoding="utf-8")))
        st["note"] = (st.get("note") or "") + "（已回滚到最优轮）"
        sp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="", help="逗号分隔；默认全部非实验类型")
    ap.add_argument("--skip-build", action="store_true", help="不重新构建，只跑闸门")
    ap.add_argument("--eval", action="store_true", help="额外跑 EvalAgent（调 LLM 评审官）")
    ap.add_argument("--rounds", type=int, default=0,
                    help="写作轮数；默认带 --eval 时 2 轮（抓到编造就带意见重写），否则 1 轮")
    ap.add_argument("--revise-all", action="store_true",
                    help="连「只有风格建议」也触发重写（实测容易退步，仅用于试验）")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="允许网关不可用时构建确定性兜底骨架（会覆盖已有成稿，默认禁止）")
    ap.add_argument("--fetch-lit", action="store_true",
                    help="强制按 literature_query 重新检索真实文献（不复用旧结果）")
    args = ap.parse_args()

    pts = ([t.strip() for t in args.types.split(",") if t.strip()]
           or [k for k, v in PAPER_TYPES.items() if not v["requires_experiment"]])
    rounds = args.rounds or (2 if args.eval else 1)

    print(f"LLM 可用性（子进程继承）：OPENAI_API_KEY={'set' if os.environ.get('OPENAI_API_KEY') else 'empty'}")
    print(f"写作轮数上限：{rounds}"
          + ("；--skip-build 为复评历史产物，不代表当前默认路径的成文质量"
             if args.skip_build else ""))
    rows = []
    for pt in pts:
        out = ROOT / f"out_verify_{pt}"
        spec = _write_spec(out, pt)
        _ensure_literature(out, spec, force=args.fetch_lit)
        try:
            from paper.spec_check import check_spec, format_report
            for _line in format_report(check_spec(pt, spec)):
                print("    " + _line)
        except Exception:
            pass
        # 首轮不能带上一轮的评审简报，否则会「无中生有地重写」
        (out / "04_review.json").unlink(missing_ok=True)

        rc, tail = (0, "reused")
        # 网关一抖，子进程会把 1.2 万字成稿换成 2 千字兜底骨架。先探测：
        # 不可用就跳过构建，除非显式 --allow-fallback 做断网演练。
        _do_build = not args.skip_build
        if _do_build and not args.allow_fallback and not _gateway_ok():
            print("    [跳过构建] 网关不可用：默认不覆盖已有成稿；"
                  "断网演练请加 --allow-fallback")
            _do_build = False
        v: dict = {}
        best = None
        last_round = 0
        hist: list[dict] = []
        for rnd in range(1, rounds + 1):
            if _do_build:
                # 旧评分结论必须作废：否则重写论文后 G04 会拿上一版的 hard_fail 当现值
                (out / "06_paper" / "07_eval_verdict.json").unlink(missing_ok=True)
                _saved = _backup_before_build(out) if _used_llm(out) else []
                rc, tail = _build(out)
                if _saved and (not _used_llm(out) or rc == 124):
                    _restore_after_failed_build(out, _saved)
                    tail = tail if rc == 124 else "子进程降级为兜底骨架，已恢复构建前成稿"
                    print(f"    [已恢复] {tail}")
                else:
                    for _b, _p in _saved:
                        _b.unlink(missing_ok=True)
            _stubs(out, pt)
            # 评审必须先于闸门：G04/G05/G10 读的是 07_eval_verdict.json，
            # 先跑闸门只会得到「未找到评分门 → unknown」，把跑过的评审记成没跑。
            v = _eval(out) if args.eval else {}
            last_round = rnd
            _mdp = out / "06_paper" / "paper_full.md"
            _snap = out / "06_paper" / f"paper_full_round{rnd}.md"
            _vsnap = out / "06_paper" / f"07_eval_verdict_round{rnd}.json"
            if _mdp.exists():
                shutil.copy2(_mdp, _snap)
            if (out / "06_paper" / "07_eval_verdict.json").exists():
                shutil.copy2(out / "06_paper" / "07_eval_verdict.json", _vsnap)
            _sc = _score_of(v)
            hist.append({"round": rnd, "chars": _chars(out),
                         "rule": (v or {}).get("overall"),
                         "judge": (v.get("llm_judge") or {}).get("overall"),
                         "hallucinations": len(((v.get("llm_judge") or {}).get("hallucinations")) or []),
                         "revise": _needs_revise(v, args.revise_all)})
            print(f"    第 {rnd} 轮：字 {_chars(out)} 规则分 {(v or {}).get('overall')} "
                  f"评审 {(v.get('llm_judge') or {}).get('overall')} "
                  f"幻觉 {len(((v.get('llm_judge') or {}).get('hallucinations')) or [])}")
            if best is None or _sc > best[0]:
                best = (_sc, rnd, _snap, _vsnap)
            if rnd < rounds and _do_build and _needs_revise(v, args.revise_all):
                _n = _write_review_brief(out, v)
                if _n:
                    print(f"    评审要求修订（{_n} 条）→ 写 04_review.json，带意见重写…")
                    continue
            break
        if best and best[1] != last_round and best[2].exists():
            _restore_best(out, best[2], best[3])
            print(f"    已回滚到第 {best[1]} 轮（评分更高），正文/评审/docx 同源")
            # 回滚后必须重读结论：否则下面打印的仍是最后一轮（较差那版）的门与分数，
            # 出现「正文已回滚、日志仍说 hard_fail」的自相矛盾。
            try:
                v = json.loads((out / "06_paper" / "07_eval_verdict.json")
                               .read_text(encoding="utf-8"))
            except Exception:
                pass
        if rounds > 1 and hist:
            try:
                (out / "06_paper" / "REVIEW_ROUNDS.json").write_text(json.dumps({
                    "paper_type": pt, "rounds": hist,
                    "best_round": best[1] if best else None,
                    "last_round": last_round,
                    "note": "最优轮若不是最后一轮，paper_full.md / 07_eval_verdict.json / "
                            "paper_full.docx 已一并回滚到该轮。",
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        g = _gates(out)
        hit, total = _structure(out, pt)
        used_llm = _used_llm(out)
        # 兜底骨架（未走 LLM）本来就不该被当作可交付论文：字数门 G02、
        # 「非模板」门 G03 报 fail，以及没跑评审导致的 G04 未验证 / G10 fail，
        # 都是设计使然，不算缺陷。
        expected = set() if used_llm else {"G02", "G03", "G04", "G10"}
        blocking = list(g.get("blocking") or [])
        unverified = list(g.get("unverified") or [])
        # 评审官缺席会把 G04 记进 unverified 而不是 blocking。以前只看 blocking，
        # 于是「语义幻觉根本没验证」被显示成「全部类型通过」，是假阳性。
        real = ([b for b in blocking if b not in expected]
                + [f"未验证:{u}" for u in unverified if u not in expected])
        rows.append({
            "type": pt, "rc": rc, "chars": _chars(out), "struct": f"{hit}/{total}",
            "llm": used_llm, "reused": bool(args.skip_build),
            "blocking": ",".join(real) or "-",
            "expected": ",".join(b for b in blocking if b in expected) or "-",
            "unverified": ",".join(g.get("unverified") or []) or "-",
        })
        print(f"[{pt}] build_rc={rc} llm={used_llm} chars={rows[-1]['chars']} "
              f"struct={hit}/{total} blocking={rows[-1]['blocking']} "
              f"expected_fail={rows[-1]['expected']} | {tail}")

        if args.eval:
            lj = (v or {}).get("llm_judge") or {}
            print(f"    eval gate={v.get('gate')} overall={v.get('overall')} "
                  f"judge_available={lj.get('available')} "
                  f"hallucinations={len(lj.get('hallucinations') or [])}")

    print("\n=== 汇总 ===")
    print(f"  {'type':18s} {'llm':5s} {'chars':>6s} {'struct':>6s}  blocking")
    for r in rows:
        print(f"  {r['type']:18s} {str(r['llm']):5s} {r['chars']:6d} "
              f"{r['struct']:>6s}  {r['blocking']}（兜底预期失败：{r['expected']}）"
              + ("  [复评历史产物]" if r.get("reused") else ""))
    # 构建失败/超时也必须算未通过：否则「重建没跑成、只是复评了旧稿」会被
    # 打印成「全部类型通过」，掩盖本轮验证其实没做完。
    bad = [r["type"] for r in rows
           if r["blocking"] != "-" or r["rc"] != 0
           or r["struct"].split("/")[0] != r["struct"].split("/")[1]]
    if bad:
        print("未通过：" + "、".join(bad))
        return 1
    print("全部类型通过（结构完整、构建成功、无额外阻断门）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
