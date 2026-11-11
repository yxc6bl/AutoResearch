"""可审计模块的配置读取器（只给 engine/provenance 等溯源模块用）。

注意：主链路的配置源是 `engine/llm_client.channels()`，支持多通道、
任意 OpenAI 兼容端点与短 Key；本文件只负责给运行清单提供环境快照，
不得用来限制主链路能接哪些 API。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
_PLACEHOLDER = re.compile(
    r"^(?:在此.*|请.*|your[_ -]?api[_ -]?key(?:[_ -]?here)?|"
    r"replace[_ -]?me|<[^>]+>|sk-your.*|changeme)$", re.I
)


def _dotenv_values(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f".env 第 {line_no} 行的配置名无效")
        values[key] = value
    return values


def load_env(root: Path = ROOT) -> dict[str, str]:
    """Process environment wins over .env; validate values before client creation."""
    values = _dotenv_values(Path(root) / ".env")
    values.update(os.environ)
    for suffix in ("", "_2", "_3"):
        for base_name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
            key = base_name + suffix
            value = values.get(key, "").strip()
            if not value:
                continue
            if any(ord(char) < 32 or ord(char) > 126 for char in value):
                raise ValueError(f"{key} 必须只含 ASCII 可打印字符，请替换 .env 中的占位值")
            if _PLACEHOLDER.fullmatch(value):
                raise ValueError(f"{key} 仍是示例占位值，请在 .env 中填入真实配置")
            # 不限制 Key 长度：本地 Ollama / 自建端点可以用短密钥
        base = values.get("OPENAI_BASE_URL" + suffix, "").strip()
        if base:
            parsed = urlsplit(base)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"OPENAI_BASE_URL{suffix} 必须是有效的 http(s) URL")
        extra = values.get("LLM_EXTRA_BODY" + suffix, "{}").strip() or "{}"
        try:
            parsed_extra = json.loads(extra)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM_EXTRA_BODY{suffix} 不是有效 JSON：{exc}") from exc
        if not isinstance(parsed_extra, dict):
            raise ValueError(f"LLM_EXTRA_BODY{suffix} 必须是 JSON 对象")
    return values


def is_real_api_key(key: str | None) -> bool:
    if not key or len(key.strip()) < 4:
        return False
    return key.isascii() and key.isprintable() and not _PLACEHOLDER.fullmatch(key.strip())
