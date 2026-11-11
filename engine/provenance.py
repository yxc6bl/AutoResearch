"""Append-only run provenance and model request records."""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from engine.config import DEFAULT_BASE_URL, DEFAULT_MODEL, load_env
from engine.io_utils import append_jsonl, atomic_write_json

SCHEMA_VERSION = 1
RUN_MANIFEST = "00_run.json"
LLM_CALL_LOG = "00_llm_calls.jsonl"
_ACTIVE_RUN_DIR = contextvars.ContextVar("autoresearch_run_dir", default=None)


def canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_paths(data) -> list[tuple[str, str]]:
    found = []

    def visit(value, key_path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                child_key = f"{key_path}.{key}" if key_path else str(key)
                lowered = str(key).lower()
                if isinstance(child, str) and (
                    lowered == "path" or lowered.endswith("_path") or lowered in {"file", "filename"}
                ):
                    found.append((child_key, child))
                else:
                    visit(child, child_key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{key_path}[{index}]")

    visit(data or {})
    return found


def _fingerprint_data(data, root: Path) -> list[dict]:
    rows, seen = [], set()
    for field, raw_path in _input_paths(data):
        if "://" in raw_path:
            rows.append({"field": field, "source": "remote", "url_sha256": hashlib.sha256(raw_path.encode()).hexdigest()})
            continue
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            cwd_candidate = Path.cwd() / candidate
            candidate = cwd_candidate if cwd_candidate.exists() else root / candidate
        resolved = candidate.resolve()
        if (field, str(resolved)) in seen:
            continue
        seen.add((field, str(resolved)))
        if not resolved.exists():
            rows.append({"field": field, "path": resolved.name, "status": "missing", "sha256": None})
        elif resolved.is_file():
            rows.append({
                "field": field, "path": resolved.name, "status": "hashed",
                "size_bytes": resolved.stat().st_size, "sha256": _sha256_file(resolved),
            })
        else:
            digest, size, count = hashlib.sha256(), 0, 0
            for child in sorted(p for p in resolved.rglob("*") if p.is_file()):
                digest.update(child.relative_to(resolved).as_posix().encode("utf-8") + b"\0")
                digest.update(_sha256_file(child).encode("ascii"))
                size += child.stat().st_size
                count += 1
            rows.append({
                "field": field, "path": resolved.name, "status": "hashed_directory",
                "file_count": count, "size_bytes": size, "sha256": digest.hexdigest(),
            })
    return rows


def _installed_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("numpy", "scikit-learn", "scipy", "pandas", "matplotlib", "openai", "python-docx"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True, timeout=2,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def active_run_dir() -> str | None:
    return _ACTIVE_RUN_DIR.get() or os.environ.get("AUTORESEARCH_RUN_DIR")


@contextmanager
def run_context(out_dir):
    previous = os.environ.get("AUTORESEARCH_RUN_DIR")
    os.environ["AUTORESEARCH_RUN_DIR"] = str(Path(out_dir).resolve())
    token = _ACTIVE_RUN_DIR.set(str(Path(out_dir).resolve()))
    try:
        yield
    finally:
        _ACTIVE_RUN_DIR.reset(token)
        if previous is None:
            os.environ.pop("AUTORESEARCH_RUN_DIR", None)
        else:
            os.environ["AUTORESEARCH_RUN_DIR"] = previous


def start_run_manifest(out_dir, spec: dict) -> dict:
    out = Path(out_dir)
    path = out / RUN_MANIFEST
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
            version = manifest.get("schema_version") if isinstance(manifest, dict) else None
            raise ValueError(f"不支持的 {RUN_MANIFEST} schema_version: {version!r}")
        if not isinstance(manifest.get("runs"), list):
            raise ValueError(f"{RUN_MANIFEST} 缺少有效的 runs 列表")
    else:
        manifest = {"schema_version": SCHEMA_VERSION, "active_run_id": None, "runs": []}

    env = load_env()
    packages = _installed_versions()
    spec_path = out / "task_spec.json"
    spec_hash = _sha256_file(spec_path) if spec_path.is_file() else hashlib.sha256(canonical_json(spec)).hexdigest()
    data = _fingerprint_data(spec.get("data"), Path(__file__).resolve().parents[1])
    llm = {
        "base_urls": list(dict.fromkeys(
            env.get("OPENAI_BASE_URL" + suffix) or (DEFAULT_BASE_URL if not suffix else "")
            for suffix in ("", "_2", "_3")
            if env.get("OPENAI_BASE_URL" + suffix) or not suffix
        )),
        "models": list(dict.fromkeys(
            env.get("OPENAI_MODEL" + suffix) or (DEFAULT_MODEL if not suffix else "")
            for suffix in ("", "_2", "_3") if env.get("OPENAI_MODEL" + suffix) or not suffix
        )),
        "temperature_source": LLM_CALL_LOG,
    }
    fingerprint_input = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": packages, "spec_sha256": spec_hash, "data": data, "llm": llm,
    }
    record = {
        "run_id": uuid.uuid4().hex,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment_revision": "autoresearch-env-v1",
        "python": fingerprint_input["python"],
        "platform": fingerprint_input["platform"],
        "packages": packages,
        "spec_sha256": spec_hash,
        "data": data,
        "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "llm": llm,
        "environment_fingerprint": hashlib.sha256(canonical_json(fingerprint_input)).hexdigest(),
    }
    manifest["runs"].append(record)
    manifest["active_run_id"] = record["run_id"]
    atomic_write_json(path, manifest)
    return record


def update_active_spec_hash(out_dir) -> None:
    path = Path(out_dir) / RUN_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"不支持的 {RUN_MANIFEST} schema_version")
    run_id = manifest.get("active_run_id")
    record = next((r for r in manifest["runs"] if r.get("run_id") == run_id), None)
    if record is None:
        raise ValueError(f"{RUN_MANIFEST} active_run_id 找不到對应运行")
    spec_path = Path(out_dir) / "task_spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"运行缺少 task_spec.json：{spec_path}")
    record["spec_sha256"] = _sha256_file(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    record["data"] = _fingerprint_data(spec.get("data"), Path(__file__).resolve().parents[1])
    packages = _installed_versions()
    env = load_env()
    llm = {
        "base_urls": list(dict.fromkeys(
            env.get("OPENAI_BASE_URL" + suffix) or (DEFAULT_BASE_URL if not suffix else "")
            for suffix in ("", "_2", "_3")
            if env.get("OPENAI_BASE_URL" + suffix) or not suffix
        )),
        "models": list(dict.fromkeys(
            env.get("OPENAI_MODEL" + suffix) or (DEFAULT_MODEL if not suffix else "")
            for suffix in ("", "_2", "_3") if env.get("OPENAI_MODEL" + suffix) or not suffix
        )),
        "temperature_source": LLM_CALL_LOG,
    }
    record["llm"] = llm
    record["packages"] = packages
    record["environment_fingerprint"] = hashlib.sha256(canonical_json({
        "python": record["python"], "platform": record["platform"],
        "packages": packages, "spec_sha256": record["spec_sha256"],
        "data": record["data"], "llm": llm,
    })).hexdigest()
    atomic_write_json(path, manifest)


def finish_run_manifest(out_dir, status: str, error: str = "") -> None:
    path = Path(out_dir) / RUN_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"不支持的 {RUN_MANIFEST} schema_version")
    run_id = manifest.get("active_run_id")
    record = next((r for r in manifest["runs"] if r.get("run_id") == run_id), None)
    if record is None:
        raise ValueError(f"{RUN_MANIFEST} active_run_id 找不到对应运行")
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    record["status"] = status
    if error:
        record["error"] = error[:1000]
    atomic_write_json(path, manifest)


def record_llm_call(model: str, temperature: float, messages, **request_params) -> None:
    out_dir = active_run_dir()
    if not out_dir:
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "request": request_params,
    }
    record["request_sha256"] = hashlib.sha256(canonical_json(record)).hexdigest()
    append_jsonl(Path(out_dir) / LLM_CALL_LOG, record)
