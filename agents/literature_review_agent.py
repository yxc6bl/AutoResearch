"""文献审评 Agent（LiteratureReviewAgent）—— 第二双眼睛审「抓来的文献」。

用户原话：光有检索 Agent 不够，还需要「另一个 Agent 去审评」——抓得好不好、哪些该用、
哪些该丢、范围够不够大。本模块即这第二道闸：

  输入：选题 + 已精选文献（01_literature.json）+ 全量候选（用于查冗余）
  输出：review verdict
    - coverage_score(1-10)：覆盖度评分
    - assessment：总体评价（中文）
    - redundant_ids：与其它文献高度重复、可剔除的
    - weak_ids：相关性弱/质量低、建议降级或剔除的
    - missing_themes：还缺哪类重要文献/主题
    - recommended_add_queries：为补强覆盖应再搜的检索式
    - keep_ratio：建议保留比例

确定性兜底：网关不可用时，用规则统计（数量、角色分布、顶刊占比、时间跨度、年份最新性）
给出覆盖分与缺口提示。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import llm_client as LLM  # noqa: E402


def _rules_review(topic: str, papers: list[dict]) -> dict:
    """确定性兜底审评。"""
    n = len(papers)
    roles = {}
    tiers = []
    years = []
    for p in papers:
        roles[p.get("role", "方法")] = roles.get(p.get("role", "方法"), 0) + 1
        tiers.append(p.get("venue_tier") or 0)
        y = p.get("year")
        if str(y).isdigit():
            years.append(int(y))
    top_ratio = (sum(1 for t in tiers if t >= 0.8) / n) if n else 0
    span = (max(years) - min(years)) if years else 0
    recent = sum(1 for y in years if y >= 2020)
    # 启发式覆盖分（1-10）
    score = 4.0
    if n >= 30:
        score += 1.5
    elif n >= 15:
        score += 0.8
    if top_ratio >= 0.3:
        score += 1.5
    elif top_ratio >= 0.15:
        score += 0.8
    if span >= 8:
        score += 1.0
    elif span >= 4:
        score += 0.5
    if recent >= 8:
        score += 1.0
    if "综述" in roles:
        score += 0.5
    gaps = []
    if n < 15:
        gaps.append(f"文献数量仅 {n} 篇，覆盖偏薄，建议扩充到 30+ 篇")
    if top_ratio < 0.15:
        gaps.append("顶刊/顶会占比过低，缺乏权威文献支撑")
    if span < 4:
        gaps.append("时间跨度不足，缺乏年代纵深")
    if "综述" not in roles:
        gaps.append("缺少综述/方法学归纳类文献，文献综述易成流水账")
    return {
        "coverage_score": round(min(10.0, score), 1),
        "assessment": f"规则兜底审评：{n} 篇文献，顶刊占比 {top_ratio:.0%}，时间跨度 {span} 年。"
                      + ("整体覆盖偏弱，建议补强。" if score < 7 else "覆盖基本合格。"),
        "redundant_ids": [],
        "weak_ids": [],
        "missing_themes": gaps,
        "recommended_add_queries": [],
        "keep_ratio": 1.0,
        "lm_driven": False,
        # 溯源：这是规则兜底，不是模型读过文献后的判断，写进台账时权重应低于人工批注。
        "provenance": "rule",
        "confidence": "medium",
    }


def run(topic: str, papers: list[dict], candidates: list[dict] | None = None) -> dict:
    """对一批精选文献做审评，返回 review verdict。"""
    if not papers:
        return {"coverage_score": 0, "assessment": "无文献可审评",
                "redundant_ids": [], "weak_ids": [], "missing_themes": ["未抓到任何文献"],
                "recommended_add_queries": [], "keep_ratio": 0.0, "lm_driven": False,
                "provenance": "rule", "confidence": "high"}
    if not LLM.available():
        return _rules_review(topic, papers)

    # 控制喂给 LM 的体量
    view = papers[:60]
    items = [{"id": p.get("id"), "title": (p.get("title") or "")[:140],
              "year": p.get("year"), "venue": (p.get("venue") or "")[:40],
              "role": p.get("role"), "q": p.get("quality_score"),
              # 检索阶段 LM 已逐篇读过：带上它的「做了什么」与「关系判断」，
              # 审评人才能判断「是否扣题」，而不是只看标题猜。
              "what": p.get("llm_what") or p.get("gist_cn") or "",
              "relation": p.get("llm_relation") or ""} for p in view]
    cand_n = len(candidates) if candidates else 0
    prompt = (
        f"研究选题：{topic}\n\n"
        f"下面是已为这篇本科毕设精选的文献（共 {len(view)} 篇 / 全量候选 {cand_n} 篇）。"
        "你是文献审评人，请独立判断这批文献是否「配得上」这个选题：\n"
        "1) coverage_score（1-10）：覆盖度与权威性是否足够；\n"
        "2) assessment：一句话总体评价；\n"
        "3) redundant_ids：与别的文献高度重复、可剔除的 id 列表；\n"
        "4) weak_ids：相关性弱或质量低、建议降级/剔除的 id 列表；\n"
        "5) missing_themes：还缺哪类重要文献/主题（如缺综述、缺最新深度学习对比、缺本数据集直接相关研究）；\n"
        "   注意：若某篇标题看着相关、但 what 显示它做的其实是另一件事，请列入 weak_ids 并说明；\n"
        "6) recommended_add_queries：为补强覆盖应再搜的检索式；\n"
        "7) keep_ratio：建议保留比例（0-1）。\n"
        '只输出 JSON：{"coverage_score":8,"assessment":"...","redundant_ids":[],'
        '"weak_ids":[],"missing_themes":["..."],"recommended_add_queries":["..."],"keep_ratio":0.9}'
    )
    try:
        out = LLM.chat_json(
            prompt + "\n\n精选文献：\n" + json.dumps(items, ensure_ascii=False),
            system=LLM.FACT_FIRST_SYSTEM, temperature=0.3, max_tokens=1800)
    except Exception as e:  # noqa: BLE001
        print(f"[ReviewAgent] LM 审评失败，转规则兜底：{e}")
        return _rules_review(topic, papers)
    if not isinstance(out, dict) or "coverage_score" not in out:
        return _rules_review(topic, papers)
    out["lm_driven"] = True
    # 模型判断：覆盖度/缺口是读出来的意见，不是硬事实，可信度标中等，
    # 后续若与人工批注冲突，台账里按 human > model > rule 排序。
    out["provenance"] = "model"
    out["confidence"] = "medium"
    # 类型兜底
    out["redundant_ids"] = out.get("redundant_ids") or []
    out["weak_ids"] = out.get("weak_ids") or []
    out["missing_themes"] = out.get("missing_themes") or []
    out["recommended_add_queries"] = out.get("recommended_add_queries") or []
    try:
        out["coverage_score"] = float(out.get("coverage_score", 5))
    except Exception:
        out["coverage_score"] = 5.0
    out["keep_ratio"] = float(out.get("keep_ratio", 1.0))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="文献审评 Agent")
    ap.add_argument("--lit", required=True, help="01_literature.json 路径")
    ap.add_argument("--topic", default="", help="研究选题")
    ap.add_argument("--cand", default="", help="02_literature_candidates.json 路径（可选）")
    args = ap.parse_args()
    papers = json.loads(Path(args.lit).read_text(encoding="utf-8")).get("papers", [])
    cands = None
    if args.cand and Path(args.cand).exists():
        cands = json.loads(Path(args.cand).read_text(encoding="utf-8"))
    print(json.dumps(run(args.topic, papers, cands), ensure_ascii=False, indent=2))
