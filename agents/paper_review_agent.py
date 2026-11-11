"""文本级论文诊断 Agent：通读全文，说清「写得好不好、哪里有问题、怎么改」。

和评分门的分工
--------------
``eval_agent`` 是给机器看的：分数、闸门、幻觉清单、结构化 issue。
本模块是给**人**看的：不给分数，不判及格线，只做指导教师式的通读诊断——
整体判断、逐章问题（必须引用原文片段）、为什么是问题、具体怎么改、按影响排序。

设计约束（用户明确说「评分没用」）：
- 输出里不出现分数、等级数字；
- 每条问题要有稳定 id（P01、P02…），人可以对某条表态「采纳/驳回」；
- ``quote`` 必须是正文里逐字出现的一段话，写不出就不许编；
- ``how`` 必须是可执行动作（补什么、删哪句、改成什么），不写「建议加强」这类空话；
- 人工批注写进 ``memory/lit_preferences.json``，下一轮写作提示会读到。

产物：``06_paper/08_paper_review.json``（机器用）与 ``08_paper_review.md``（给人读）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import llm_client as LLM  # noqa: E402

PROMPT = """你是本科毕业论文的评阅专家，手上有清北等高校高质量本科论文的参照标尺（见后）。
请通读下面这篇论文，做**产出质量诊断**，不要打分、不要给等级。

你的立场（重要）：规则系统已经查过引用编号、章节齐全、篇幅、零编造这些「有没有」，
那不是你的工作。你只判断「写得好不好」——论证是否成立、证据是否支撑结论、
文献是否真的被用起来、有没有独立见解、章节是否真的承担了论证任务、文字是否可读。
不要因为模板字段齐全就判为高质量，也不要拿规则分数当依据。

写成一份能让学生照着改的诊断意见，输出严格 JSON：

{
  "verdict": "可直接提交 / 修改后可提交 / 需要较大修改 / 不建议提交 四选一",
  "overall": "300 字以内：这篇论文整体是什么水平、最影响质量的 2-3 件事是什么、能不能交。要具体，不要套话。",
  "quality_vs_reference": "200 字以内：与参照范文相比，本篇在论证深度、证据链、原创性、文献运用上最实质的 2-3 个差距；讲内容，不要罗列数字或模板字段",
  "strengths": ["确实做得好的地方（2-4 条）"],
  "chapters": [
    {"chapter": "第N章 章名",
     "judgment": "这一章写得怎么样、承担的作用有没有做到",
     "problems": [
       {"quote": "必须是论文中逐字出现的一段原文（20-60 字），用来定位问题",
        "why": "为什么这里有问题：论证、证据、逻辑、表述上的具体毛病",
        "how": "具体怎么改：补什么内容、删哪句、改成什么说法、移到哪一节。必须是能直接动手的动作",
        "severity": "高 / 中 / 低"}
     ]}
  ],
  "cross_cutting": ["跨章节的问题，如术语不一致、章节小结与正文不符、引用密度失衡、前后结论矛盾"],
  "top_actions": [
    {"priority": 1, "action": "最该先做的一件事（动作句）", "where": "落在哪一章/哪一节", "why": "为什么优先"}
  ],
  "suggested_queries": ["若发现文献覆盖不足，给出 2-4 条可直接用于 arXiv/OpenAlex 的英文检索式；没有文献问题则给空数组"]
}

硬性要求：
1. 不许出现任何分数、百分号、等级数字；这是文字诊断，不是评分表；
2. quote 必须能在正文逐字找到，找不到依据就写别的句子，绝不编造；
3. how 不许写「建议加强」「应进一步提升」这类空话，要写成可执行的动作；
4. 每章都给 judgment；确实没问题就给空 problems 并说明为什么；
5. top_actions 最多 8 条，按对论文质量的影响从大到小排，优先写能立刻动手的；
6. suggested_queries 必须是简短的英文检索短语（不要中文长句、不要整段解释），
   它们会被存进记忆，下一轮抓文献时并入检索计划；
7. 只输出 JSON，不要解释、不要 markdown 代码块；
8. quality_vs_reference 必须结合后面给出的范文标尺来写，不许写「建议对标优秀论文」这类空话；
9. 若正文是模板拼接稿（句子缺主语、中英混杂、研究问题被原样塞进摘要、文献流水账），
   必须在 overall 里直接点名，这是最影响产出质量的问题，优先级高于任何结构建议。
"""


def _reference_context(md: str) -> str:
    """把本地范文的参照标尺 + 本篇确定性差距拼进评审上下文。

    这是「学习高质量论文」的落点：不训练模型，而是把清北范文的可观测质量
    特征整理成专家评阅的参照系，让模型按同一把尺子判断产出质量。
    """
    try:
        import benchmark_corpus as BC
        corpus = str(ROOT / "corpus")
        ref = BC.reference_profile(corpus)
        gap = BC.quality_gap(md, corpus)
        return ("\n\n" + ref["rubric_text"] + "\n\n" + gap
                + "\n\n【提醒】标尺里的量只是参照，最终判断要落到论证与文字本身。\n")
    except Exception as e:  # noqa: BLE001
        return f"\n\n（范文参照库不可用：{e}；请仅依据正文做质量判断）\n"



def _fallback(md: str) -> dict:
    """网关不可用时的确定性兜底：只报告数得出来的文本事实，明确标注未完成通读。

    实测踩过两个坑：把「封面 / 诚信声明 / 英文 Abstract」当成展开不足来报；
    用汉字数衡量英文段落，得到「0 个汉字」的假问题。所以这里只检查正文
    「第N章」章节，并用总字符数（含英文与标点）衡量篇幅。
    """
    chapters = [l.strip() for l in md.splitlines() if re.match(r"^##\s", l)]
    secs = re.split(r"^##\s+", md, flags=re.M)[1:]
    _skip = ("封面", "诚信声明", "目录", "摘要", "abstract", "参考文献",
             "致谢", "附录", "声明", "原创", "keywords", "关键词")
    thin = []
    for s in secs:
        name = s.splitlines()[0].strip()[:40]
        if any(k in name.lower() for k in _skip):
            continue
        if not re.match(r"^第\s*\d+\s*章", name):
            continue
        n = len(re.sub(r"\s", "", s))
        if n < 500:
            thin.append(f"「{name}」正文仅 {n} 个字符，展开不足")
    problems = []
    for i, t in enumerate(thin[:6], 1):
        problems.append({"id": f"P{i:02d}", "quote": "", "why": t,
                         "how": "按该章应承担的论证任务补足实质内容，而不是加套话",
                         "severity": "中"})
    try:
        import benchmark_corpus as BC
        _gap = BC.quality_gap(md, str(ROOT / "corpus"))
    except Exception:
        _gap = "（范文对标不可用）"
    return {
        "verdict": "未完成文本诊断（网关不可用）",
        "overall": "本次网关不可用，模型没有通读全文，因此不给出整体质量判断。"
                   f"下面只列能确定性数出的事实：共 {len(chapters)} 个二级章节、"
                   f"正文引用 {len(re.findall(r'\[\d+\]', md))} 处。"
                   "请在网关恢复后重跑本步骤，才能得到逐章问题与具体改法。",
        "strengths": [], "chapters": [], "cross_cutting": thin,
        "top_actions": [{"id": p["id"], "priority": i, "action": p["how"],
                         "where": "", "why": p["why"]}
                        for i, p in enumerate(problems, 1)],
        "problems": problems,
        "quality_gap": _gap,
        "provenance": "rule", "llm_available": False,
    }


def run(paper_path, out_dir=None, verdict: dict | None = None) -> dict:
    """通读论文，写 08_paper_review.json / .md，返回诊断结果。"""
    p = Path(paper_path)
    md = p.read_text(encoding="utf-8")
    out = Path(out_dir) if out_dir else p.parent
    out.mkdir(parents=True, exist_ok=True)

    # 已有的机器评审结论只作为线索，不代替通读；人要看的是文本本身。
    hints = ""
    if verdict:
        lj = verdict.get("llm_judge") or {}
        bits = (verdict.get("issues") or [])[:4] + (lj.get("hallucinations") or [])[:3]
        if bits:
            hints = "\n\n【系统已自动查出的线索（仍请自己通读核对）】\n" + "\n".join(
                f"- {str(x)[:160]}" for x in bits)

    reference = _reference_context(md)
    review = None
    if LLM.available():
        full = PROMPT + reference + hints + "\n\n【论文全文】\n" + md
        tight = ("\n\n⚠️ 上一次输出不是合法 JSON。请只输出一个 JSON 对象，不要解释、"
                 "不要 markdown 代码块；每章最多 2 条 problems，top_actions 最多 6 条，"
                 "每条文字不超过 100 字。")
        for i in range(2):
            try:
                obj = LLM.chat_json(full + (tight if i else ""),
                                    system=LLM.FACT_FIRST_SYSTEM,
                                    temperature=0.25,
                                    max_tokens=(3600 if i == 0 else 4000))
            except Exception:
                obj = {}
            if isinstance(obj, dict) and obj.get("overall"):
                review = obj
                break
    if review is None:
        review = _fallback(md)
        review["llm_available"] = False
    else:
        review["provenance"] = "model"
        review["llm_available"] = True

    # 确定性护栏：quote 必须逐字出现在正文；编造的定位句直接清掉。
    dropped = 0
    for ch in review.get("chapters") or []:
        kept = []
        for pr in ch.get("problems") or []:
            q = str(pr.get("quote") or "").strip()
            if q and q not in md:
                dropped += 1
                continue
            kept.append(pr)
        ch["problems"] = kept
    # 稳定编号：逐章顺序 + 跨章问题之后统一编 P01…，人按这个编号表态。
    counter = 0
    for ch in review.get("chapters") or []:
        for pr in ch.get("problems") or []:
            counter += 1
            pr["id"] = f"P{counter:02d}"
    actions = review.get("top_actions") or []
    for i, a in enumerate(actions, 1):
        a.setdefault("priority", i)
        a.setdefault("id", f"A{i:02d}")
    review["dropped_unverifiable_quotes"] = dropped
    review["paper"] = str(p)
    review["ts"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # 自我反馈闭环：诊断发现的文献缺口 -> 下轮检索式，写进跨运行记忆。
    # 只有模型真通读过才存，规则兜底不配给检索建议（会污染计划）。
    try:
        from engine import selection_ledger as SL
        if review.get("llm_available"):
            review["feedback_queries_saved"] = SL.add_feedback_queries(
                review.get("suggested_queries") or [], source="paper_review",
                note=f"论文诊断：{str(review.get('verdict') or '')}")
        else:
            review["feedback_queries_saved"] = 0
        # 本次成文的检索计划里，有哪些式子来自上一轮反馈——让闭环可被看见。
        from engine import lit_plan as LP
        _plan = LP.load(out.parent) or LP.load(out)
        review["applied_feedback"] = (_plan or {}).get("applied_feedback") or []
    except Exception:
        review["feedback_queries_saved"] = 0
        review["applied_feedback"] = []

    # 专家意见回流：写入反馈台账，并挂到 04_review.json 供下一轮写作读取。
    # 规则兜底只记确定性差距，不冒充模型通读的结论。
    try:
        from engine import feedback as FB
        _run_dir = str(out.parent)
        review["feedback_records_saved"] = FB.record_expert_review(
            review, run_dir=_run_dir,
            topic=(verdict or {}).get("topic") or "",
            review_brief=str(Path(_run_dir) / "04_review.json"))
    except Exception:
        review["feedback_records_saved"] = 0
    review["reference_profile_used"] = True

    review["human_feedback_hint"] = (
        "对某条意见表态：python auto_research.py lit-pref --review-mark "
        "<产物目录> P01 accepted --note \"…\"；采纳/驳回会存进记忆并影响下一轮写作。")

    (out / "08_paper_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "08_paper_review.md").write_text(_render(review), encoding="utf-8")
    return review


def _render(r: dict) -> str:
    """渲染成给人读的 Markdown 诊断书。"""
    L = [f"# 论文文本诊断", "",
         f"- 结论：**{r.get('verdict') or '未给出'}**",
         f"- 生成时间：{r.get('ts')}",
         f"- 诊断方式：{'模型通读全文' if r.get('llm_available') else '规则兜底（未经模型通读）'}",
         f"- 无法核实的定位句：{r.get('dropped_unverifiable_quotes', 0)} 条（已删除）", "",
         "## 整体判断", "", r.get("overall") or "（无）", ""]
    if r.get("quality_vs_reference"):
        L += ["## 与高质量范文的实质差距", "", r["quality_vs_reference"], ""]
    if r.get("quality_gap"):
        L += ["## 范文对标（确定性信号）", "", r["quality_gap"], ""]
    if r.get("strengths"):
        L += ["## 做得好的地方", ""] + [f"- {x}" for x in r["strengths"]] + [""]
    L += ["## 逐章诊断", ""]
    if not (r.get("chapters") or []):
        L += ["（本次未完成逐章诊断；网关恢复后重跑本步骤）", ""]
    for ch in r.get("chapters") or []:
        L += [f"### {ch.get('chapter') or '（未命名章节）'}", "",
              ch.get("judgment") or "（未给出判断）", ""]
        for pr in ch.get("problems") or []:
            L += [f"**{pr.get('id')}｜严重程度：{pr.get('severity') or '中'}**", ""]
            if pr.get("quote"):
                L += [f"> 原文：{pr['quote']}", ""]
            L += [f"- 问题：{pr.get('why') or ''}", "",
                  f"- 怎么改：{pr.get('how') or ''}", ""]
        if not (ch.get("problems") or []):
            L += ["（本章未发现需要改动的具体问题）", ""]
    if r.get("cross_cutting"):
        L += ["## 跨章节问题", ""] + [f"- {x}" for x in r["cross_cutting"]] + [""]
    if r.get("applied_feedback"):
        L += ["## 本次检索已应用的历史反馈", "",
              "下面这几条检索式是上一轮诊断／审评留下的，本次抓文献时已并入计划：", ""]
        for x in r["applied_feedback"]:
            L.append(f"- `{x.get('query')}`（来源：{x.get('source') or '未知'}"
                     + (f"；{x.get('note')}" if x.get("note") else "") + "）")
        L.append("")
    if r.get("top_actions"):
        L += ["## 优先修改清单", ""]
        for a in sorted(r["top_actions"], key=lambda x: x.get("priority") or 99):
            L += [f"{a.get('priority')}. **{a.get('id')}** {a.get('action') or ''}",
                  f"   - 位置：{a.get('where') or '见对应章节'}",
                  f"   - 为什么先改：{a.get('why') or ''}", ""]
    L += ["## 怎么反馈", "", r.get("human_feedback_hint") or "", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="论文文本诊断：不打分，只说问题与改法")
    ap.add_argument("paper", help="paper_full.md 路径")
    ap.add_argument("--out", default="", help="输出目录（默认与论文同目录）")
    ap.add_argument("--verdict", default="", help="07_eval_verdict.json 路径（可选线索）")
    a = ap.parse_args()
    v = None
    if a.verdict and Path(a.verdict).exists():
        try:
            v = json.loads(Path(a.verdict).read_text(encoding="utf-8"))
        except Exception:
            v = None
    r = run(a.paper, a.out or None, v)
    print(json.dumps({"verdict": r.get("verdict"),
                      "n_chapters": len(r.get("chapters") or []),
                      "n_problems": sum(len(c.get("problems") or [])
                                        for c in r.get("chapters") or []),
                      "n_actions": len(r.get("top_actions") or []),
                      "llm_available": r.get("llm_available"),
                      "dropped_quotes": r.get("dropped_unverifiable_quotes")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
