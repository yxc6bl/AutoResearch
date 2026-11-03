"""文献子代理（LiteratureAgent）v2。

职责（I/O 契约见 docs/FRAMEWORK.md）：
    输入：query, max_results, sources, top_n, selection(可选), store, out_path, source_tag
    输出：本次用于论文的「已选」文献列表

相较 v1 的改进（对应导师「抓太少 / 质量不高 / 给使用者选择权」）：
    1. 多源抓取：arxiv + openalex（免费、无需 Key、覆盖远超 arXiv）
    2. 质量评分：用被引数 + 时效性算 quality_score，候选按分排序
    3. 选择窗口：所有候选先落 MemoryStore 与 02_literature_candidates.json，
       再由 select() 选定——默认 auto 取 top_n；后续网页界面将把候选渲染为
       勾选框，把 chosen_ids 传回即实现「使用者自己挑」。

记忆价值：抓过的文献（含来源、质量分）跨运行去重累积；被哪些产出引用记在 cited_in，
供后续偏好建模与「记住抓过什么文献」使用。
"""
from __future__ import annotations

import json
import math
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# 保证从任意入口加载时都能 import 同级模块
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.store import MemoryStore, store as default_store  # noqa: E402
import literature as lit_mod  # noqa: E402


# ── 网络工具 ──
def _ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _reconstruct_abstract(inv) -> str:
    """OpenAlex 用倒排索引存摘要，还原成普通文本。"""
    if not inv:
        return ""
    pos = []
    for word, locs in inv.items():
        for p in locs:
            pos.append((p, word))
    pos.sort()
    return " ".join(w for _, w in pos)


# ── OpenAlex 源（免费、无需 Key，覆盖期刊/会议，含被引数） ──
OPENALEX = "https://api.openalex.org/works"


# ── 期刊/会议档次（venue tier）：文献质量的核心信号 ──
# 之前只按「被引数 + 时效性」打分，会把最新、最相关的论文（被引尚低）压到 0.3 左右，
# 导致「平均质量分 0.32、高质量仅 1 篇」。引入同行评审的顶会/顶刊档次信号，
# 让真正高水准确认过的文献在选文献时优先浮出。
# 用正则词边界 + 负向预查，避免「Cell Biology」「computer science」等子串误判为顶刊。
# 注意：原白名单过度偏向「生医 + 机器学习会议」，导致航空/机械/声学等领域期刊被错判 tier 0；
# 现扩展工程/航空/声学等领域 A 档，并增加「DOI 前缀 → 出版社档次」兜底，覆盖任意学科。
_VENUE_ASTAR_RE = [
    r"\b(neurips|nips|icml|iclr|cvpr|iccv|eccv|kdd|aaai|ijcai|miccai)\b",
    r"\bieee tpami\b", r"\bjmlr\b", r"medical image analysis",
    r"\bnature\b", r"\bscience\b(?! of)", r"\bcell\b(?! biology)",
    r"\blancet\b", r"\bnejm\b", r"\bjama\b",
    r"international conference on machine learning",
]
_VENUE_A_RE = [
    # 生医信息
    r"ieee jbhi", r"ieee transactions on medical imaging",
    r"artificial intelligence in medicine", r"computers in biology and medicine",
    r"bmc medical informatics", r"npj digital medicine", r"plos medicine",
    r"plos computational biology", r"bioinformatics",
    r"journal of biomedical informatics", r"health informatics journal",
    # 工程 / 航空 / 声学 / 机械 / 流体
    r"procedia", r"engineering analysis with boundary elements",
    r"journal of sound and vibration", r"applied acoustics",
    r"physics of fluids", r"aerospace science and technology",
    r"mechanical systems and signal processing", r"journal of fluids engineering",
    r"journal of computational physics", r"applied energy",
    r"journal of the acoustical society", r"aerospace", r"applied sciences",
    r"entropy", r"machines", r"processes", r"computation",
    # 综合 / 计算机
    r"scientific reports", r"sensors", r"ieee access",
    r"lecture notes in computer science", r"lecture notes in networks and systems",
]

# DOI 前缀 → 出版社档次兜底（覆盖任意学科，避免领域期刊被错判 tier 0）
# 学会/顶刊出版社给更高档；普通商业/MDPI 给中档；未知出版社给 0。
_PUBLISHER_TIER = {
    "10.1016": 0.6,    # Elsevier
    "10.1007": 0.6,    # Springer
    "10.1109": 0.7,    # IEEE
    "10.1145": 0.7,    # ACM
    "10.1002": 0.6,    # Wiley
    "10.1088": 0.6,    # IOP
    "10.1115": 0.7,    # ASME（机械/流体）
    "10.2514": 0.7,    # AIAA（航空）
    "10.1177": 0.55,   # SAGE
    "10.1080": 0.55,   # Taylor & Francis
    "10.1021": 0.6,    # ACS
    "10.1063": 0.6,    # AIP
    "10.1155": 0.5,    # Hindawi
    "10.1504": 0.5,    # Inderscience
    "10.3390": 0.5,    # MDPI
    "10.1038": 0.8,    # Nature 子刊
}


def _doi_prefix(url: str) -> str:
    if not url:
        return ""
    m = re.search(r"doi\.org/([^/]+)/", str(url))
    return m.group(1) if m else ""


def _venue_tier(venue: str, url: str = "") -> float:
    """返回 1.0（顶会/顶刊）/0.8（强刊）/0.5~0.7（已知出版社期刊）/0.0（未知或预印本）。"""
    v = (venue or "").lower()
    if v and any(re.search(p, v) for p in _VENUE_ASTAR_RE):
        return 1.0
    if v and any(re.search(p, v) for p in _VENUE_A_RE):
        return 0.8
    # 即便 venue 字段为空/未知，也尝试从 DOI 前缀判断出版社档次
    prefix = _doi_prefix(url)
    if prefix in _PUBLISHER_TIER:
        return _PUBLISHER_TIER[prefix]
    return 0.0


# ── 参考文献角色分类（用于综述按路线组织 + 保证正文全覆盖引用） ──
_ROLE_SURVEY = ("survey", "review", "systematic review", "state-of-the-art", "taxonomy", "literature review")
_ROLE_DATASET = ("dataset", "benchmark", "corpus", "database", "data set")
_ROLE_THEORY = ("theory", "theoretical", "framework", "foundation", "principle", "formalism")


def _classify_role(p: dict) -> str:
    """把文献归类为 综述 / 基准数据集 / 理论基础 / 方法 之一（依据标题关键词）。"""
    t = (p.get("title") or "").lower()
    if any(k in t for k in _ROLE_SURVEY):
        return "综述"
    if any(k in t for k in _ROLE_DATASET):
        return "基准数据集"
    if any(k in t for k in _ROLE_THEORY):
        return "理论基础"
    return "方法"


def fetch_openalex(query: str, max_results: int = 10, timeout: int = 30,
                  max_retries: int = 3) -> list[dict]:
    params = {
        "search": query,
        "per_page": max(50, int(max_results) * 3),
        "sort": "relevance_score:desc",
        "select": "id,title,abstract_inverted_index,authorships,primary_location,"
                  "publication_year,cited_by_count,doi,relevance_score",
    }
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            url = OPENALEX + "?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(
                url, headers={"User-Agent": "AutoResearch/1.2 (literature fetch)"})
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as r:
                data = json.loads(r.read().decode("utf-8", "ignore"))
            out = []
            for w in data.get("results", []):
                doi = w.get("doi")
                oid = w.get("id", "")
                key = ("doi:" + doi) if doi else ("openalex:" + oid.rsplit("/", 1)[-1])
                authors = ", ".join(
                    a.get("author", {}).get("display_name", "")
                    for a in w.get("authorships", []) if a.get("author", {}).get("display_name")
                )
                src = w.get("primary_location", {}).get("source") or {}
                venue = src.get("display_name") or "OpenAlex"
                out.append({
                    "id": key,
                    "title": (w.get("title") or "").strip(),
                    "authors": authors,
                    "venue": venue,
                    "year": str(w.get("publication_year") or ""),
                    "url": doi or oid,
                    "abstract": _reconstruct_abstract(w.get("abstract_inverted_index"))[:700],
                    "cited_by_count": w.get("cited_by_count") or 0,
                    "relevance_score": w.get("relevance_score") or 0.0,
                })
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < max_retries:
                time.sleep(1.2 * attempt)
    raise RuntimeError(f"OpenAlex 检索失败（已重试 {max_retries} 次）：{last!r}")


# ── Semantic Scholar 源（可选，默认关闭） ──
# 覆盖带期刊/会议元数据的正式论文 + 引用量，弥补「基本是预印本」的短板，进一步「扩大范围」。
# 该 API 公共端点限流较严（无 Key 约 1 req/10s），故默认不启用；设置环境变量
# AUTORESEARCH_USE_SEMSCHOLAR=1 才接入。失败时静默跳过，绝不拖垮主流程。
SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/search"


def fetch_semantic_scholar(query: str, max_results: int = 10, timeout: int = 12,
                           max_retries: int = 1) -> list[dict]:
    """从 Semantic Scholar Graph API 拉取正式论文（含 venue / 引用量）。失败抛异常由调用方吞掉。"""
    params = {
        "query": query,
        "limit": min(100, max(10, int(max_results) * 2)),
        "fields": "title,abstract,year,venue,authors,citationCount,externalIds,url",
    }
    url = SEMANTIC_SCHOLAR + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "AutoResearch/1.2 (literature fetch)"})
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as r:
                data = json.loads(r.read().decode("utf-8", "ignore"))
            out = []
            for p in data.get("data", []):
                doi = (p.get("externalIds") or {}).get("DOI")
                oid = p.get("url") or ""
                key = ("doi:" + doi) if doi else ("s2:" + (oid.rsplit("/", 1)[-1] or p.get("title", "")))
                authors = ", ".join(
                    a.get("name", "") for a in (p.get("authors") or []) if a.get("name"))
                out.append({
                    "id": key,
                    "title": (p.get("title") or "").strip(),
                    "authors": authors,
                    "venue": (p.get("venue") or "").strip() or "Semantic Scholar",
                    "year": str(p.get("year") or ""),
                    "url": doi or oid,
                    "abstract": (p.get("abstract") or "")[:700],
                    "cited_by_count": p.get("citationCount") or 0,
                    "relevance_score": 0.0,
                })
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < max_retries:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"Semantic Scholar 检索失败：{last!r}")


# ── 质量评分（venue 档次为核心信号） ──
def _cite_score(cites: int, year: int) -> float:
    """被引数按「每年被引」对数归一，避免新论文因被引少被系统性压低。"""
    if not cites:
        return 0.0
    cur = datetime.now().year
    age = max(1, cur - year) if year else cur
    cpy = cites / max(1, age)
    return min(1.0, math.log10(1 + cpy) / math.log10(1 + 25))


def _quality_score(p: dict) -> float:
    tier = _venue_tier(p.get("venue"), p.get("url"))
    cites = p.get("cited_by_count", 0) or 0
    cite_score = _cite_score(cites, int(p.get("year") or 0))
    try:
        year = int(p.get("year") or 0)
    except Exception:
        year = 0
    recency = max(0.0, min(1.0, (year - 2010) / 16.0)) if year else 0.3
    rel = p.get("relevance_score") or 0.0
    rel_score = min(1.0, rel / 30.0) if rel else 0.3
    # 权重：档次 0.45 > 被引 0.25 > 时效 0.15 > 相关性 0.15
    return round(0.45 * tier + 0.25 * cite_score + 0.15 * recency + 0.15 * rel_score, 3)


# ── 相关性过滤：剔除跑题噪声（如检索「翼型噪声」却返回板球/假新闻等） ──
_STOP = {"machine", "learning", "deep", "neural", "using", "based", "study", "studies",
         "analysis", "approach", "model", "models", "data", "prediction", "regression",
         "the", "a", "an", "of", "with", "and", "on", "via", "from", "for", "to", "in",
         "network", "networks", "method", "methods", "using", "based", "recent", "advances"}


def _sig_tokens(query: str) -> list[str]:
    toks = re.findall(r"[a-z0-9\-]+", (query or "").lower())
    return [t for t in toks if len(t) > 2 and t not in _STOP]


def _relevance_passes(p: dict, query: str) -> bool:
    """候选是否贴题：标题/摘要命中任一查询关键术语，或 OpenAlex 相关性分极高。"""
    sig = _sig_tokens(query)
    if not sig:
        return True
    text = ((p.get("title") or "") + " " + (p.get("abstract") or "")).lower()
    if any(t in text for t in sig):
        return True
    rs = p.get("relevance_score") or 0
    if rs >= 25:   # OpenAlex 高相关性兜底，即使无关键词字面命中
        return True
    return False


# ── 合并 / 去重 / 评分 ──
def fetch(query: str, max_results: int = 10,
          sources: tuple = ("arxiv", "openalex")) -> list[dict]:
    """多源抓取并合并去重，返回按 quality_score 降序的候选列表。

    内部对 OpenAlex 多拉取（更宽网 + 过滤），arXiv 维持原量（多为预印本噪声，
    经相关性过滤后仅保留真正贴题者）。
    """
    raw = []
    if "arxiv" in sources:
        try:
            raw += lit_mod.fetch_arxiv(query, max_results=max_results)
        except Exception as e:  # noqa: BLE001
            print(f"[LiteratureAgent] arXiv 失败，跳过：{e}")
    if "openalex" in sources:
        try:
            raw += fetch_openalex(query, max_results=max(20, int(max_results) * 2))
        except Exception as e:  # noqa: BLE001
            print(f"[LiteratureAgent] OpenAlex 失败，跳过：{e}")
    # 相关性过滤：剔除跑题噪声（领域无关，靠查询关键术语判定）
    before = len(raw)
    raw = [p for p in raw if _relevance_passes(p, query)]
    if before:
        print(f"[LiteratureAgent] 相关性过滤：{before} → {len(raw)} 篇（剔除跑题 {before - len(raw)}）")
    seen = {}
    for p in raw:
        pid = p.get("id") or p.get("url")
        if not pid or pid in seen:
            continue
        p["source"] = "arxiv" if "arxiv.org" in str(p.get("url", "")) else "openalex"
        p["venue_tier"] = _venue_tier(p.get("venue"), p.get("url"))
        p["role"] = _classify_role(p)
        p["quality_score"] = _quality_score(p)
        seen[pid] = p
    return sorted(seen.values(), key=lambda x: x.get("quality_score", 0), reverse=True)


# ── 选择窗口 ──
def select(candidates: list[dict], mode: str = "auto", top_n: int = 40,
           chosen_ids: list[str] | None = None) -> list[dict]:
    """从候选中选定用于论文的文献（流水线默认 auto，不阻塞）。

    选择策略（对齐「文献数 ↑ / 顶刊占比 ↑ / 角色覆盖」三项短板）：
      1) 先确保顶刊/顶会配额：quality_score 排序后，优先取 venue_tier≥0.8 的前
         max(10, top_n//3) 篇，保证顶会/顶刊来源占比；
      2) 再用剩余候选按质量分填满至 top_n；
      3) chosen_ids 给定时按使用者勾选返回（网页界面将走这条路）。
    """
    if chosen_ids is not None:
        idset = set(chosen_ids)
        return [c for c in candidates if c["id"] in idset]
    for c in candidates:
        c.setdefault("venue_tier", _venue_tier(c.get("venue"), c.get("url")))
        c.setdefault("role", _classify_role(c))
    ranked = sorted(candidates, key=lambda x: x.get("quality_score", 0), reverse=True)
    venue_quota = max(10, top_n // 3)
    chosen, seen_ids = [], set()
    for c in ranked:
        if c.get("venue_tier", 0) >= 0.8 and \
           sum(1 for x in chosen if x.get("venue_tier", 0) >= 0.8) < venue_quota:
            chosen.append(c)
            seen_ids.add(c["id"])
        if len(chosen) >= top_n:
            break
    for c in ranked:
        if c["id"] in seen_ids:
            continue
        chosen.append(c)
        seen_ids.add(c["id"])
        if len(chosen) >= top_n:
            break
    return chosen


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _store_candidate(store: MemoryStore, c: dict, tag: str) -> None:
    rec = dict(c)
    rec["fetched_at"] = _now_iso()
    rec.setdefault("source_query", "")
    rec.setdefault("cited_in", [tag])
    store.put("literature", c["id"], rec)


def run(query: str, max_results: int = 10, sources: tuple = ("arxiv", "openalex"),
        store: MemoryStore | None = None, out_path: Path | None = None,
        source_tag: str = "", top_n: int = 40,
        selection: list[str] | None = None) -> list[dict]:
    """执行一次文献获取：多源抓取 → 落记忆 → 选择 → 写产物。

    返回本次用于论文的「已选」文献列表。
    """
    store = store or default_store
    tag = source_tag or "manual"

    # 1) 已记忆的（跨运行去重）
    all_ids = set(store.list("literature"))
    existing_same_q = store.query("literature", {"source_query": query}) if query else []

    # 2) 多源抓取候选
    candidates = fetch(query, max_results=max_results, sources=sources)

    # 3) 全部候选落 MemoryStore（记忆层累积；已有则补 cited_in）
    merged = list(existing_same_q)
    seen_this_run = {p.get("id") for p in existing_same_q}
    for c in candidates:
        pid = c.get("id") or c.get("url") or c.get("title", "")
        if pid in all_ids:
            rec = store.get("literature", pid) or {}
            cited = set(rec.get("cited_in", []))
            cited.add(tag)
            rec["cited_in"] = sorted(cited)
            store.put("literature", pid, rec)
        else:
            c["source_query"] = query
            _store_candidate(store, c, tag)
            all_ids.add(pid)
        if pid not in seen_this_run:
            merged.append(c)
            seen_this_run.add(pid)

    # 4) 排序后选择
    merged.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    chosen = select(merged, mode="auto", top_n=top_n, chosen_ids=selection)

    # 5) 写产物（论文用）+ 候选清单（使用者「窗口」原料）
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"papers": chosen}, ensure_ascii=False, indent=2), encoding="utf-8")
        cand_path = out_path.parent / "02_literature_candidates.json"
        cand_path.write_text(json.dumps(
            [{"id": c["id"], "title": c.get("title"), "venue": c.get("venue"),
              "year": c.get("year"), "source": c.get("source"),
              "quality_score": c.get("quality_score"), "url": c.get("url")}
             for c in merged], ensure_ascii=False, indent=2), encoding="utf-8")
    return chosen
