"""Durable helpers for audit-critical run artifacts."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

_APPEND_LOCK = threading.Lock()


def atomic_write_text(path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path, value) -> None:
    target = Path(path)
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with _APPEND_LOCK:
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("short write while appending JSONL record")
                offset += written
            os.fsync(fd)
        finally:
            os.close(fd)
