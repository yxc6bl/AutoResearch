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


def fetch_openalex(query: str, max_results: int = 10, timeout: int = 30,
                  max_retries: int = 3) -> list[dict]:
    params = {
        "search": query,
        "per_page": max_results,
        "sort": "relevance_score:desc",
        "select": "id,title,abstract_inverted_index,authorships,primary_location,"
                  "publication_year,cited_by_count,doi",
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
                })
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < max_retries:
                time.sleep(1.2 * attempt)
    raise RuntimeError(f"OpenAlex 检索失败（已重试 {max_retries} 次）：{last!r}")


# ── 质量评分 ──
def _quality_score(p: dict) -> float:
    cites = p.get("cited_by_count", 0) or 0
    cite_score = min(1.0, cites / 100.0)
    try:
        year = int(p.get("year") or 0)
    except Exception:
        year = 0
    recency = max(0.0, min(1.0, (year - 2010) / 16.0)) if year else 0.3
    return round(0.6 * cite_score + 0.4 * recency, 3)


# ── 合并 / 去重 / 评分 ──
def fetch(query: str, max_results: int = 10,
          sources: tuple = ("arxiv", "openalex")) -> list[dict]:
    """多源抓取并合并去重，返回按 quality_score 降序的候选列表。"""
    raw = []
    if "arxiv" in sources:
        try:
            raw += lit_mod.fetch_arxiv(query, max_results=max_results)
        except Exception as e:  # noqa: BLE001
            print(f"[LiteratureAgent] arXiv 失败，跳过：{e}")
    if "openalex" in sources:
        try:
            raw += fetch_openalex(query, max_results=max_results)
        except Exception as e:  # noqa: BLE001
            print(f"[LiteratureAgent] OpenAlex 失败，跳过：{e}")
    seen = {}
    for p in raw:
        pid = p.get("id") or p.get("url")
        if not pid or pid in seen:
            continue
        p["source"] = "arxiv" if "arxiv.org" in str(p.get("url", "")) else "openalex"
        p["quality_score"] = _quality_score(p)
        seen[pid] = p
    return sorted(seen.values(), key=lambda x: x.get("quality_score", 0), reverse=True)


# ── 选择窗口 ──
def select(candidates: list[dict], mode: str = "auto", top_n: int = 8,
           chosen_ids: list[str] | None = None) -> list[dict]:
    """从候选中选定用于论文的文献。
    mode="auto"：取 quality_score 最高的 top_n（流水线默认，不阻塞）。
    chosen_ids 给定：按使用者勾选返回（网页界面将走这条路）。
    """
    if chosen_ids is not None:
        idset = set(chosen_ids)
        return [c for c in candidates if c["id"] in idset]
    return candidates[:top_n]


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
        source_tag: str = "", top_n: int = 8,
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
