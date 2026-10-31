"""文献子代理（LiteratureAgent）。

职责（I/O 契约见 docs/FRAMEWORK.md）：
    输入：query(str), max_results(int), store(MemoryStore), out_path(可选), source_tag(可选)
    输出：本次用于论文的文献列表（dict 列表）

流程：
    1. 先查 MemoryStore 的 literature/ 命名空间（跨运行去重缓存）
    2. 从 arXiv 抓取本次 query 的文献（失败则回退到已有记忆，不阻塞主流程）
    3. 抓到的按 id 去重写回 MemoryStore（合并 cited_in，记 fetched_at / source_query）
    4. 同时写 out_path/01_literature.json 以兼容论文生成器

记忆价值：导师要求的「记住抓过什么文献」在此落地——同一篇文献跨多次运行
只会存一次，且记录它被哪些产出目录引用过（cited_in），供后续偏好建模使用。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# 保证从任意入口加载时都能 import 同级模块
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.store import MemoryStore, store as default_store  # noqa: E402
import literature as lit_mod  # noqa: E402


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def run(query: str, max_results: int = 10, store: MemoryStore | None = None,
        out_path: Path | None = None, source_tag: str = "") -> list[dict]:
    """执行一次文献获取并落记忆。返回本次用于论文的文献列表。"""
    store = store or default_store
    tag = source_tag or "manual"

    # 1) 已记忆的文献（全局按 id 去重）
    all_ids = set(store.list("literature"))
    existing_same_q = store.query("literature", {"source_query": query}) if query else []

    # 2) 抓取（失败不阻塞：返回已有记忆）
    try:
        fresh = lit_mod.fetch_arxiv(query, max_results=max_results)
    except Exception as e:  # noqa: BLE001
        print(f"[LiteratureAgent] arXiv 抓取失败，回退到已有记忆：{e}")
        fresh = []

    # 3) 写回 MemoryStore（去重 + 合并 cited_in）
    merged = list(existing_same_q)
    seen_this_run = {p.get("id") for p in existing_same_q}
    for p in fresh:
        pid = p.get("id") or p.get("url")
        if not pid:
            continue
        if pid in all_ids:
            # 已记忆：仅补充 cited_in（本次产出目录）
            rec = store.get("literature", pid) or {}
            cited = set(rec.get("cited_in", []))
            cited.add(tag)
            rec["cited_in"] = sorted(cited)
            store.put("literature", pid, rec)
            continue
        rec = dict(p)
        rec["id"] = pid
        rec["fetched_at"] = _now_iso()
        rec["source_query"] = query
        rec.setdefault("cited_in", [tag])
        store.put("literature", pid, rec)
        all_ids.add(pid)
        if pid not in seen_this_run:
            merged.append(rec)
            seen_this_run.add(pid)

    # 4) 兼容层：写 01_literature.json 供论文生成器读取
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"papers": merged}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return merged
