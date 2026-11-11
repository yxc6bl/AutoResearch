"""Version checks for persisted task specifications."""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1


def validate_task_spec(value, source="task_spec.json") -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{source} 必须是 JSON 对象")
    version = value.get("schema_version")
    if version is not None and (type(version) is not int or version != SCHEMA_VERSION):
        raise ValueError(f"不支持的 {source} schema_version: {version!r}")
    return value


def load_task_spec(path) -> dict:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取任务规格 {source}: {exc}") from exc
    return validate_task_spec(value, source.name)
