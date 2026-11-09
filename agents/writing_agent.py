#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WritingAgent 薄壳（子代理拆分之一）。

封装 paper/generate_paper.py 的调用（subprocess），把本次论文产物路径写入
MemoryStore 的 project 命名空间（"paper"）。

预留 style_profile 读取接口：组员完成风格蒸馏后，将 style 画像写入 MemoryStore
的 style 命名空间（"profile"），届时本 agent 读取并透传给 generate_paper 以
实现「反 AI 腔 / 像人写」（需 generate_paper 增加 --style-json 支持，TODO）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.store import MemoryStore, store as default_store  # noqa: E402


def _env():
    """继承当前环境，并在缺失 API Key 时从 .env 补充（供独立调用场景）。"""
    env = os.environ.copy()
    dot = ROOT / ".env"
    if "OPENAI_API_KEY" not in env and dot.exists():
        try:
            for line in dot.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
        except Exception:
            pass
    return env


def run(out_dir, task_spec_path, store: MemoryStore | None = None, timeout: int = 1800):
    """调用 generate_paper 生成论文；返回 {paper, paper_md, returncode, stderr}。"""
    store = store or default_store
    out_dir = Path(out_dir)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "paper" / "generate_paper.py"),
         "--output-dir", str(out_dir), "--task-spec", str(task_spec_path)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=timeout,
        env=_env(),
    )
    paper = out_dir / "06_paper" / "paper_full.docx"
    paper_md = out_dir / "06_paper" / "paper_full.md"
    store.put("project", "paper", {
        "out_dir": str(out_dir),
        "paper_docx": str(paper) if paper.exists() else None,
        "paper_md": str(paper_md) if paper_md.exists() else None,
    })
    # 生成器会把写作参与情况写入 06_paper/WRITE_STATUS.json，并在 stdout 打一行
    # @@WRITE_STATUS@@ 哨兵。优先读文件（最可靠），其次读哨兵行，最后才退回整段反解。
    status_path = out_dir / "06_paper" / "WRITE_STATUS.json"
    write_status: dict = {}
    if status_path.exists():
        try:
            write_status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            write_status = {}
    llm_written = write_status.get("llm_written")
    if llm_written is None:
        for _line in (proc.stdout or "").splitlines():
            if _line.startswith("@@WRITE_STATUS@@"):
                try:
                    write_status = json.loads(_line[len("@@WRITE_STATUS@@"):].strip())
                    llm_written = write_status.get("llm_written")
                except Exception:
                    pass
                break
    if llm_written is None:
        # 兼容旧版生成器：整段 stdout 末尾的 JSON
        try:
            _out = proc.stdout or ""
            _start = _out.rfind("{")
            if _start >= 0:
                llm_written = json.loads(_out[_start:]).get("llm_written")
        except Exception:
            llm_written = None

    # 成文过程的两个关键信号必须让主流程看见，否则素材缺口与重写都在静默中发生：
    #   [素材]           —— 素材完备度（缺什么、哪些偏薄）
    #   [generate_paper —— 第几批带评审意见重写
    # 其余日志仍然不转发，避免把 run 的控制台刷成一屏。
    for _line in (proc.stdout or "").splitlines():
        if _line.startswith("[素材]") or _line.startswith("[generate_paper·"):
            print(_line)

    return {
        "paper": str(paper) if paper.exists() else None,
        "paper_md": str(paper_md) if paper_md.exists() else None,
        "returncode": proc.returncode,
        "llm_written": llm_written,
        "write_status": write_status,
        "stderr": (proc.stderr or "")[-2500:],
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="写作 Agent：产物目录 → 论文（可单独调用）")
    ap.add_argument("--dir", required=True, help="产物目录（需含 task_spec.json 等）")
    ap.add_argument("--timeout", type=int, default=1800, help="子进程超时（秒）")
    a = ap.parse_args()
    _d = Path(a.dir)
    _r = run(_d, _d / "task_spec.json", timeout=a.timeout)
    print(json.dumps(_r, ensure_ascii=False, indent=2))
    raise SystemExit(0 if _r["returncode"] == 0 else 1)
