"""Versioned, hash-checked storage for facts shared by writing and review."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from engine.io_utils import atomic_write_json

SCHEMA_VERSION = 1


class FactPackError(ValueError):
    pass


def _digest(facts: str) -> str:
    return hashlib.sha256(facts.encode("utf-8")).hexdigest()


def _read_document(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FactPackError(f"无法读取 FACT_PACK.json：{exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        version = document.get("schema_version") if isinstance(document, dict) else None
        raise FactPackError(f"不支持的 FACT_PACK.json schema_version: {version!r}")
    if not isinstance(document.get("versions"), list):
        raise FactPackError("FACT_PACK.json 缺少有效的 versions 列表")
    return document


def write_fact_pack(path, *, paper_type: str, facts: str, metadata: dict | None = None) -> dict:
    target = Path(path)
    digest = _digest(facts)
    document = _read_document(target) if target.exists() else {
        "schema_version": SCHEMA_VERSION, "active_sha256": None, "versions": []
    }
    record = next((row for row in document["versions"] if row.get("sha256") == digest), None)
    if record:
        if record.get("facts") != facts:
            raise FactPackError("FACT_PACK.json 中同一 sha256 对应不同事实内容")
    else:
        record = {
            "sha256": digest,
            "paper_type": paper_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "facts": facts,
        }
        document["versions"].append(record)
    document["active_sha256"] = digest
    atomic_write_json(target, document)
    return record


def read_fact_pack(path, expected_sha256: str | None = None) -> dict:
    target = Path(path)
    if not target.is_file():
        raise FactPackError("FACT_PACK.json 不存在")
    document = _read_document(target)
    digest = expected_sha256 or document.get("active_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise FactPackError("FACT_PACK.json 的 active_sha256 无效")
    record = next((row for row in document["versions"] if row.get("sha256") == digest), None)
    if not isinstance(record, dict) or not isinstance(record.get("facts"), str):
        raise FactPackError(f"FACT_PACK.json 找不到 sha256={digest}")
    if _digest(record["facts"]) != digest:
        raise FactPackError(f"FACT_PACK.json 哈希校验失败：{digest}")
    return record
