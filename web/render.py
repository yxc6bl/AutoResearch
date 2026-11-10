"""AutoResearch 网页渲染模块（零外部依赖）。

负责：
- md_to_html：把论文 Markdown 渲染为带样式的 HTML（章/节标题、加粗、引用角标、目录）。
- verdict_to_html：把 EvalAgent 的评分 verdict 渲染为评分卡片（六维彩条 + 硬门徽章）。
- build_standalone：生成一份自包含预览 HTML（图表以 base64 内嵌），点开即看。
- find_paper_dir / collect_result：从一次运行产物目录里收集论文/评分/文献/实验/图表。
"""
from __future__ import annotations

import base64
import html
import json
import os
import re
from pathlib import Path

# ── Markdown → HTML（极简、够用，不依赖第三方） ──────────────────────

def _inline(text: str) -> str:
    """行内格式：加粗 **x**、引用 [n]、转义 HTML。"""
    text = html.escape(text, quote=False)
    # [1][2] 引用 → 角标
    text = re.sub(r"\[(\d+)\]", r'<sup class="cite">[\1]</sup>', text)
    # **粗体**
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return text


def md_to_html(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    para: list[str] = []
    toc_open = False   # 有未闭合的 <ul class="toc">
    list_buf: list[str] = []
    list_tag = "ul"

    def flush_list():
        nonlocal list_tag
        if list_buf:
            out.append(f'<{list_tag} class="md-list">' + "".join(list_buf) + f"</{list_tag}>")
            list_buf.clear()
        list_tag = "ul"

    def flush_para():
        if para:
            joined = " ".join(x.strip() for x in para if x.strip())
            if joined:
                out.append(f"<p>{_inline(joined)}</p>")
            para.clear()

    def close_toc():
        nonlocal toc_open
        if toc_open:
            out.append("</ul>")
            toc_open = False

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_para()
            flush_list()
            continue
        if line.startswith("### "):
            flush_para()
            flush_list()
            close_toc()
            out.append(f'<h3>{_inline(stripped[4:])}</h3>')
        elif line.startswith("## "):
            flush_para()
            flush_list()
            close_toc()
            title = stripped[3:].strip()
            if title == "目录":
                toc_open = True
                out.append('<h2 class="toc-title">目录</h2>')
                out.append('<ul class="toc">')
            else:
                cls = "chapter" if re.match(r"^第[一二三四五六七八九十\d]+章", title) else "section-h2"
                out.append(f'<h2 class="{cls}">{_inline(title)}</h2>')
        elif line.startswith("# "):
            flush_para()
            flush_list()
            close_toc()
            out.append(f'<h1 class="doc-title">{_inline(stripped[2:])}</h1>')
        elif stripped.startswith("> "):
            flush_para()
            flush_list()
            close_toc()
            out.append(f'<blockquote>{_inline(stripped[2:])}</blockquote>')
        elif toc_open:
            # 目录区块一直延续到下一个标题；空行不该把它截断
            out.append(f'<li>{_inline(stripped)}</li>')
        elif re.match(r"^[-*] ", stripped):
            flush_para()
            if list_tag != "ul":
                flush_list()
            list_tag = "ul"
            list_buf.append(f'<li>{_inline(stripped[2:])}</li>')
        elif re.match(r"^\d+[.、] ", stripped):
            flush_para()
            if list_tag != "ol":
                flush_list()
            list_tag = "ol"
            body = re.sub(r"^\d+[.、]\s*", "", stripped)
            list_buf.append(f'<li>{_inline(body)}</li>')
        else:
            para.append(line)
    flush_para()
    flush_list()
    close_toc()
    return "\n".join(out)


# ── 评分 verdict → 卡片 HTML ──────────────────────────────────────────

_DIM_COLOR = {
    "topic_sig": "#6a1b9a", "literature": "#ad1457", "structure": "#1565c0",
    "methodology": "#2e7d32", "innovation": "#ef6c00", "writing_norms": "#00838f",
    "consistency": "#444",
}
_DIM_NAME = {
    "topic_sig": "选题意义", "literature": "文献调研", "structure": "逻辑构建",
    "methodology": "专业能力·方法内容", "innovation": "创新", "writing_norms": "撰写质量与规范",
    "consistency": "零编造硬门", "authenticity": "真实性", "anti_ai": "反 AI 痕迹",
    "content_quality": "内容质量", "mathematical_rigor": "数学严谨",
}

def _score_list(v: dict) -> list[tuple[str, float]]:
    """把 scores 字段统一成 [(中文名, 分数)]。

    先按标准 rubric 顺序排，再接上本版 rubric 里的其他维度——旧产物用的是
    authenticity / anti_ai / content_quality 这套键，硬编码顺序会把它们整段漏掉。
    """
    sc = v.get("scores") or {}
    if not isinstance(sc, dict):
        return []
    order = ["topic_sig", "literature", "structure", "methodology", "innovation",
             "writing_norms", "consistency"]
    keys = [k for k in order if k in sc]
    keys += [k for k in sc if k not in order and k != "overall"]
    out = []
    for k in keys:
        try:
            out.append((_DIM_NAME.get(k, k), float(sc[k])))
        except (TypeError, ValueError):
            continue
    return out

def verdict_to_html(v: dict) -> str:
    overall = v.get("overall", 0)
    gate = v.get("gate", "")
    passed = v.get("passed", False)
    lit_grade = v.get("literature_grade", "")
    grade = v.get("grade", "")
    issues = v.get("issues", []) or []

    badge = (
        '<span class="badge ok">✔ 通过 · 硬门 ' + html.escape(str(gate)) + "</span>"
        if passed else
        '<span class="badge bad">✘ 未通过 · 建议重做</span>'
    )
    grade_badge = f'<span class="badge grade">{html.escape(str(grade))}</span>' if grade else ""
    lit_badge = f'<span class="badge lit">文献评级 {html.escape(str(lit_grade))}</span>' if lit_grade else ""

    bars = []
    for name, sc in _score_list(v):
        color = _DIM_COLOR.get(name, "#555")
        pct = max(0, min(100, int(sc)))
        bars.append(f"""
        <div class="dim">
          <div class="dim-head"><span>{html.escape(str(name))}</span><b>{sc}</b></div>
          <div class="bar"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>
        </div>""")

    issue_html = ""
    if issues:
        lis = "".join(f"<li>{html.escape(str(x))}</li>" for x in issues[:8])
        issue_html = f'<div class="issues"><b>待改进 ({len(issues)})</b><ul>{lis}</ul></div>'

    return f"""
    <div class="verdict-card">
      <div class="verdict-top">
        <div class="score-big">{overall}</div>
        <div class="verdict-meta">
          <div class="verdict-label">综合质量评分</div>
          <div class="verdict-badges">{badge}{grade_badge}{lit_badge}</div>
        </div>
      </div>
      <div class="dims">{''.join(bars)}</div>
      {issue_html}
    </div>"""


# ── 产物收集 ──────────────────────────────────────────────────────────

def find_paper_dir(base: Path) -> Path | None:
    """在运行产物根目录里定位包含 paper_full.md 的 06_paper 目录。"""
    base = Path(base)
    candidates = []
    direct = base / "06_paper"
    if (direct / "paper_full.md").exists():
        candidates.append(direct)
    # iterate 模式：round_N/06_paper
    for rd in sorted(base.glob("round_*"), reverse=True):
        pd = rd / "06_paper"
        if (pd / "paper_full.md").exists():
            candidates.append(pd)
    if not candidates:
        return None
    # 选论文最长的那份（内容最完整）
    return max(candidates, key=lambda p: (p / "paper_full.md").stat().st_size)


ROOT = Path(__file__).resolve().parent.parent


def _rel(p: Path) -> str:
    """转成相对工程根的 posix 路径（供 /api/file 解析）。"""
    try:
        return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return ""


def _read_json(p) -> dict | None:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def _paper_stats(md: str) -> dict:
    """正文统计：汉字数、段落数、引用处数、章节数。只数不猜。"""
    han = len(re.findall(r"[\u4e00-\u9fff]", md))
    paras = [ln for ln in md.split("\n") if ln.strip() and not ln.lstrip().startswith(("#", ">", "-", "*", "|"))]
    cites = re.findall(r"\[(\d+)\]", md)
    chapters = re.findall(r"^##\s*第[一二三四五六七八九十\d]+章", md, flags=re.M)
    return {
        "chars": han,
        "paragraphs": len(paras),
        "citations": len(cites),
        "distinct_refs": len(set(cites)),
        "chapters": len(chapters),
    }


def collect_extras(base: Path, pd: Path) -> dict:
    """补齐「论文之外」的可展示证据：诊断、闸门、写作状态、选材台账、可下载产物。"""
    extras: dict = {}

    # 评审 Agent 的文本诊断
    review_md = pd / "08_paper_review.md"
    extras["review_html"] = md_to_html(review_md.read_text(encoding="utf-8")) if review_md.exists() else ""
    extras["review"] = _read_json(pd / "08_paper_review.json") or {}

    # 质量闸门
    gates = _read_json(base / "07_gate_report.json") or {}
    extras["gates"] = {
        "decision": gates.get("decision") or {},
        "counts": gates.get("counts") or {},
        "gates": [
            {"id": g.get("id"), "level": g.get("level"), "name": g.get("name"),
             "stage": g.get("stage"), "status": g.get("status"),
             "rule": g.get("rule"), "evidence": g.get("evidence"),
             "provenance": g.get("provenance")}
            for g in (gates.get("gates") or [])
        ],
    } if gates else {}

    # 写作与评审轮次
    ws = _read_json(pd / "WRITE_STATUS.json") or {}
    extras["write"] = {
        "llm_written": bool(ws.get("llm_written")),
        "model": ws.get("model") or "",
        "paper_type": ws.get("paper_type") or "",
        "sections": len(ws.get("llm_sections") or []),
    } if ws else {}
    extras["rounds"] = _read_json(pd / "REVIEW_ROUNDS.json") or {}

    # 选材台账
    led = _read_json(base / "LIT_SELECTION.json") or {}
    if led:
        decs = led.get("decisions") or []
        by: dict[str, int] = {}
        for d in decs:
            by[str(d.get("by") or "rule")] = by.get(str(d.get("by") or "rule"), 0) + 1
        extras["selection"] = {
            "n_candidates": led.get("n_candidates"),
            "n_selected": led.get("n_selected"),
            "n_extra_from_refetch": led.get("n_extra_from_refetch"),
            "selector": led.get("selector") or "",
            "preferences_applied": led.get("preferences_applied") or {},
            "review": led.get("review") or {},
            "by_actor": by,
            "n_human": sum(1 for d in decs if d.get("by") == "human"),
        }
    # 检索计划（留痕）
    plan = _read_json(base / "LIT_PLAN.json") or {}
    if plan:
        st = plan.get("strategy") or {}
        extras["plan"] = {
            "sources": st.get("sources") or [],
            "query_groups": st.get("query_groups") or [],
            "applied_feedback": plan.get("applied_feedback") or [],
            "outcome": plan.get("outcome") or {},
            "topic_reading": plan.get("topic_reading") or "",
        }

    # 可下载产物
    arts = []
    for name, label, kind in (
        ("paper_full.docx", "Word 文档", "docx"),
        ("paper_full.md", "Markdown 正文", "md"),
        ("08_paper_review.md", "文本诊断", "md"),
        ("07_eval_verdict.json", "评分明细", "json"),
    ):
        f = pd / name
        if f.exists():
            arts.append({"label": label, "kind": kind, "rel": _rel(f),
                         "size": f.stat().st_size, "name": name})
    gr = base / "07_gate_report.json"
    if gr.exists():
        arts.append({"label": "闸门报告", "kind": "json", "rel": _rel(gr),
                     "size": gr.stat().st_size, "name": gr.name})
    extras["artifacts"] = arts
    return extras


def collect_result(base: str) -> dict:
    """从一次运行的产物目录收集所有可展示内容。"""
    base = Path(base)
    pd = find_paper_dir(base)
    res: dict = {
        "paper_html": "", "verdict": None, "literature": [],
        "experiment": None, "plots": [], "out_dir": str(base),
        "found": False,
    }
    if pd is None:
        return res
    res["found"] = True

    md = (pd / "paper_full.md").read_text(encoding="utf-8")
    res["paper_html"] = md_to_html(md)
    res["stats"] = _paper_stats(md)
    res["paper_dir_rel"] = _rel(pd)

    vp = pd / "07_eval_verdict.json"
    if vp.exists():
        try:
            res["verdict"] = json.loads(vp.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 文献候选：02_literature_candidates.json（优先）或 01_literature.json
    for name in ("02_literature_candidates.json", "01_literature.json"):
        lp = base / name
        if lp.exists():
            try:
                data = json.loads(lp.read_text(encoding="utf-8"))
                cands = data if isinstance(data, list) else (data.get("papers") or data.get("candidates") or [])
                res["literature"] = cands[:60]
                break
            except Exception:
                pass

    ep = base / "03_experiment.json"
    if not ep.exists():
        for rd in sorted(base.glob("round_*"), reverse=True):
            if (rd / "03_experiment.json").exists():
                ep = rd / "03_experiment.json"
                break
    if ep.exists():
        try:
            res["experiment"] = json.loads(ep.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 图表：plots 目录
    plot_dirs = [base / "plots"]
    for rd in sorted(base.glob("round_*"), reverse=True):
        plot_dirs.append(rd / "plots")
    for pd_ in plot_dirs:
        if pd_.is_dir():
            for f in sorted(pd_.glob("*.png")):
                # 相对工程根的路径，供 /api/file 解析
                res["plots"].append(str(f))

    res.update(collect_extras(base, pd))
    return res


# ── 独立预览页（图表 base64 内嵌，点开即看） ──────────────────────────

_PREVIEW_CSS = """
:root{--ink:#1a1a1a;--muted:#666;--line:#e3e3e3;--bg:#fff;--accent:#1565c0;}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:var(--ink);background:#f5f6f8;line-height:1.8}
.wrap{max-width:920px;margin:0 auto;padding:32px 22px 80px}
.doc-title{font-size:24px;font-weight:800;text-align:center;margin:8px 0 24px}
h2.chapter{font-size:20px;font-weight:700;margin:34px 0 10px;padding-left:12px;border-left:5px solid var(--accent)}
h2.section-h2{font-size:18px;font-weight:700;margin:28px 0 8px}
h3{font-size:15px;font-weight:600;margin:18px 0 6px;color:#333}
p{margin:10px 0;text-align:justify}
sup.cite{color:var(--accent);font-size:11px;font-weight:600;padding:0 1px}
strong{font-weight:700}
.toc-title{font-size:16px;font-weight:700;margin:24px 0 6px}
ul.toc{list-style:none;padding-left:0;columns:2;column-gap:30px}
ul.toc li{padding:3px 0;color:#444;font-size:14px;border-bottom:1px dotted #ddd}
blockquote{background:#f0f4fb;border-left:4px solid var(--accent);margin:12px 0;padding:8px 14px;color:#444}
.verdict-card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin:22px 0;box-shadow:0 2px 10px rgba(0,0,0,.05)}
.verdict-top{display:flex;align-items:center;gap:20px;margin-bottom:16px}
.score-big{font-size:46px;font-weight:800;color:var(--accent);min-width:90px;text-align:center}
.verdict-label{font-size:13px;color:var(--muted)}
.verdict-badges{margin-top:6px;display:flex;gap:8px;flex-wrap:wrap}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:13px;font-weight:600}
.badge.ok{background:#e6f4ea;color:#1e7e34}
.badge.bad{background:#fdecea;color:#c0392b}
.badge.lit{background:#fce4ec;color:#ad1457}
.badge.grade{background:#e8eaf6;color:#283593;font-weight:700}
.dims{display:grid;grid-template-columns:1fr 1fr;gap:10px 24px}
.dim-head{display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px}
.dim-head b{color:var(--accent)}
.bar{height:8px;background:#eee;border-radius:6px;overflow:hidden}
.bar-fill{height:100%;border-radius:6px}
.issues{margin-top:14px;font-size:13px;color:#b45309}
.issues ul{margin:6px 0 0;padding-left:18px}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:22px 0}
.plots img{width:100%;border:1px solid var(--line);border-radius:10px;background:#fff;padding:8px}
.lit-item{padding:10px 0;border-bottom:1px solid var(--line);font-size:13.5px}
.lit-item .t{font-weight:600}
.lit-item .m{color:var(--muted);font-size:12px}
footer{text-align:center;color:#aaa;font-size:12px;margin-top:40px}
"""

def build_standalone(base: str, out_html: str) -> str:
    """生成自包含预览 HTML（图表 base64 内嵌）。返回写出路径。"""
    data = collect_result(base)
    if not data["found"]:
        raise FileNotFoundError(f"在 {base} 未找到 paper_full.md")

    plots_html = ""
    if data["plots"]:
        cells = []
        for p in data["plots"]:
            try:
                b64 = base64.b64encode(Path(p).read_bytes()).decode()
                cells.append(f'<div><img src="data:image/png;base64,{b64}" alt="plot"></div>')
            except Exception:
                pass
        if cells:
            plots_html = f'<h2 class="section-h2">图表</h2><div class="plots">{"".join(cells)}</div>'

    lit_html = ""
    if data["literature"]:
        items = []
        for c in data["literature"][:20]:
            title = c.get("title", "(无标题)")
            meta = " · ".join(str(x) for x in [c.get("authors"), c.get("venue"), c.get("year")] if x)
            items.append(f'<div class="lit-item"><div class="t">{html.escape(str(title))}</div>'
                         f'<div class="m">{html.escape(str(meta))}</div></div>')
        lit_html = f'<h2 class="section-h2">参考文献（{len(data["literature"])} 篇）</h2>{"".join(items)}'

    verdict_html = verdict_to_html(data["verdict"]) if data["verdict"] else ""

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoResearch 论文预览</title><style>{_PREVIEW_CSS}</style></head>
<body><div class="wrap">
{verdict_html}
{data["paper_html"]}
{plots_html}
{lit_html}
<footer>由 AutoResearch 多智能体论文系统生成 · 自包含离线预览</footer>
</div></body></html>"""

    Path(out_html).write_text(doc, encoding="utf-8")
    return out_html
