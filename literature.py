"""文献层（AutoResearch 子模块）。

目标：让论文的「相关工作」引用**真实可检索**的文献，而不是占位空壳。

实现：调用 arXiv 公开 API（http://export.arxiv.org/api/query），
该接口**无需任何 API Key**，仅依赖标准库 urllib + xml。
按研究主题检索若干论文，解析 Atom 后写入 `01_literature.json`，
格式与论文生成器 build_references() 完全兼容。

用法：
    # 作为库
    from literature import build_literature
    papers = build_literature("random forest diabetes prediction", "out/01_literature.json")

    # 命令行
    python literature.py "random forest diabetes prediction" --out out/ --max 8
"""
from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
# 使用 HTTPS（443）。部分受限 / 沙箱环境会拦截明文 HTTP（80）出站，
# 切换到 HTTPS 通常即可绕过；实测 export.arxiv.org:443 可达。
ENDPOINT = "https://export.arxiv.org/api/query"


def _ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _parse_entries(data: str) -> list[dict]:
    root = ET.fromstring(data)
    papers: list[dict] = []
    for e in root.findall(f"{ATOM}entry"):
        title = (e.findtext(f"{ATOM}title") or "").strip().replace("\n", " ")
        summary = (e.findtext(f"{ATOM}summary") or "").strip().replace("\n", " ")
        published = (e.findtext(f"{ATOM}published") or "").strip()
        year = published[:4] if published else ""
        authors = [
            (a.findtext(f"{ATOM}name") or "").strip()
            for a in e.findall(f"{ATOM}author")
            if (a.findtext(f"{ATOM}name") or "").strip()
        ]
        aid = (e.findtext(f"{ATOM}id") or "").strip()
        cat = e.find(f"{ARXIV}primary_category")
        venue = "arXiv:" + cat.get("term") if cat is not None else "arXiv"
        abs_url = aid
        if abs_url and "/abs/" not in abs_url:
            abs_url = aid.replace("http://arxiv.org/", "https://arxiv.org/abs/")
        papers.append({
            "title": title,
            "authors": ", ".join(authors),
            "venue": venue,
            "year": year,
            "url": abs_url,
            "id": abs_url,
            "abstract": summary[:700],
        })
    return papers


def fetch_arxiv(query: str, max_results: int = 8, timeout: int = 30, max_retries: int = 3) -> list[dict]:
    """检索 arXiv 并返回论文列表（无需 Key）。

    端点为 HTTPS（443）。部分受限 / 沙箱环境会拦截明文 HTTP（80）出站，
    切换到 HTTPS 一般即可绕过。请求失败会指数退避重试，全部失败则抛出
    带端点与底层异常信息的 RuntimeError，便于排查（而非静默返回空）。
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            url = ENDPOINT + "?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(
                url, headers={"User-Agent": "AutoResearch/1.1 (automated literature fetch)"}
            )
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as r:
                if getattr(r, "status", 200) != 200:
                    raise urllib.error.HTTPError(url, r.status, r.reason, r.headers, None)
                data = r.read().decode("utf-8", "ignore")
            return _parse_entries(data)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(
        f"arXiv 文献检索失败（已重试 {max_retries} 次，端点 {ENDPOINT}）：{last_err!r}"
    )


def build_literature(query: str, out_path, max_results: int = 8) -> list[dict]:
    """检索文献并写入 out_path（01_literature.json）。返回论文列表。"""
    papers = fetch_arxiv(query, max_results=max_results)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"papers": papers}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return papers


def main() -> int:
    ap = argparse.ArgumentParser(description="从 arXiv 检索真实文献并写入 01_literature.json（无需 API Key）")
    ap.add_argument("query", help="检索词（建议英文关键词，越具体越好）")
    ap.add_argument("--out", default="01_literature.json", help="输出 JSON 路径，默认当前目录 01_literature.json")
    ap.add_argument("--max", type=int, default=8, help="检索条数，默认 8")
    args = ap.parse_args()
    papers = build_literature(args.query, args.out, max_results=args.max)
    print(f"检索到 {len(papers)} 篇文献，已写入 {args.out}")
    for i, p in enumerate(papers[: args.max], 1):
        print(f"  [{i}] {p['title'][:70]}  ({p['year']}, {p['venue']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
