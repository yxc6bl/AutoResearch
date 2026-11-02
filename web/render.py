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
    toc_active = False

    def flush_para():
        if para:
            joined = " ".join(x.strip() for x in para if x.strip())
            if joined:
                out.append(f"<p>{_inline(joined)}</p>")
            para.clear()

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_para()
            toc_active = False
            continue
        if line.startswith("### "):
            flush_para()
            toc_active = False
            out.append(f'<h3>{_inline(stripped[4:])}</h3>')
        elif line.startswith("## "):
            flush_para()
            title = stripped[3:].strip()
            if title == "目录":
                toc_active = True
                out.append('<h2 class="toc-title">目录</h2>')
                out.append('<ul class="toc">')
            else:
                if toc_active:  # 目录区块结束
                    out.append("</ul>")
                    toc_active = False
                cls = "chapter" if re.match(r"^第[一二三四五六七八九十\d]+章", title) else "section-h2"
                out.append(f'<h2 class="{cls}">{_inline(title)}</h2>')
        elif line.startswith("# "):
            flush_para()
            if toc_active:
                out.append("</ul>")
                toc_active = False
            out.append(f'<h1 class="doc-title">{_inline(stripped[2:])}</h1>')
        elif stripped.startswith("> "):
            flush_para()
            out.append(f'<blockquote>{_inline(stripped[2:])}</blockquote>')
        elif re.match(r"^[-*] ", stripped):
            flush_para()
            out.append(f'<li>{_inline(stripped[2:])}</li>')
        elif toc_active:
            out.append(f'<li>{_inline(stripped)}</li>')
        else:
            para.append(line)
    flush_para()
    if toc_active:
        out.append("</ul>")
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
    "consistency": "零编造硬门",
}

def _score_list(v: dict) -> list[tuple[str, float]]:
    """把 scores 字段统一成 [(中文名, 分数)]，按 rubric 维度顺序排列。"""
    sc = v.get("scores") or {}
    if isinstance(sc, dict):
        order = ["topic_sig", "literature", "structure", "methodology", "innovation", "writing_norms", "consistency"]
        return [(_DIM_NAME.get(k, k), float(sc[k])) for k in order if k in sc]
    return []

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
                res["literature"] = cands[:30]
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
