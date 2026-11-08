#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GitHub 同步：配置远端 / 提交并推送 / 提交后自动同步。

设计约束（都是踩过的坑）：

* Token 绝不落盘。不写进 remote URL、不写 .git/config、不写 hook 文件；
  只在 push 时用 `git -c http.extraheader=...` 临时注入当前进程。
  长期凭据交给 Git Credential Manager（本机已装 2.9.0），第一次推送弹窗授权一次即可。
* post-commit hook 只做「推送」，绝不再 commit，否则会递归触发自己。
* hook 里用解释器绝对路径，Git 的 PATH 里未必有 python。
* 同步失败不阻塞提交、不打印到用户脸上，日志写 .git/sync.log（.git 不入库）。

用法：
    python scripts/github_sync.py --setup https://github.com/OWNER/REPO.git
    python scripts/github_sync.py                      # add -A + commit + push
    python scripts/github_sync.py -m "修了某处"        # 指定提交信息
    python scripts/github_sync.py --status             # 看远端/领先落后
    python scripts/github_sync.py --auto               # hook 内部调用，只推不提交
    GITHUB_TOKEN=ghp_xxx python scripts/github_sync.py --store-token
        # 把 PAT 存进系统凭据管理器（GCM 弹窗不可用时用），之后免密
    python scripts/github_sync.py --no-hook            # 不装自动推送 hook
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / ".git" / "sync.log"


def git(*args: str, extra_config: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = ["git", "-C", str(ROOT)]
    for c in extra_config or []:
        cmd += ["-c", c]
    cmd += list(args)
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def log(msg: str) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")
    except Exception:
        pass


def _strip_token(url: str) -> str:
    """如果用户把 token 直接贴进了 URL，剥掉再写进 .git/config。"""
    return re.sub(r"(https?://)[^/@]+@", r"\1", url)


def _token_args() -> list[str]:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    tok = tok.strip()
    if not tok:
        return []
    # 只作用于本次 push 进程，不会写进任何文件
    return [f"http.extraheader=AUTHORIZATION: bearer {tok}"]


def setup(url: str, install_hook: bool = True) -> int:
    clean = _strip_token(url)
    if not re.match(r"^(https://github\.com/|git@github\.com:)", clean):
        print(f"[warn] 看起来不是 GitHub 地址：{clean}（仍会尝试配置）")
    cur = git("remote", "get-url", "origin")
    if cur.returncode == 0 and cur.stdout.strip():
        print(f"[setup] 替换已有 origin：{cur.stdout.strip()} → {clean}")
        r = git("remote", "set-url", "origin", clean)
    else:
        r = git("remote", "add", "origin", clean)
    if r.returncode != 0:
        print("[error] 配置 origin 失败：", r.stderr.strip())
        return 1
    # 仓库级凭据助手，不动全局配置；GCM 会把授权存在 Windows 凭据管理器里
    git("config", "--local", "credential.helper", "manager")
    git("config", "--local", "push.default", "simple")
    git("config", "--local", "user.name", "Yang Xincheng")
    git("config", "--local", "user.email", "yangxincheng@users.noreply.github.com")
    print(f"[setup] origin = {clean}")
    print("[setup] 凭据交给 Git Credential Manager（首次推送会弹窗授权一次）")

    if install_hook:
        hook = ROOT / ".git" / "hooks" / "post-commit"
        hook.parent.mkdir(parents=True, exist_ok=True)
        py = sys.executable.replace("\\", "/")
        script = (Path(__file__).resolve()).as_posix()
        hook.write_text(
            "#!/bin/sh\n"
            "# 由 scripts/github_sync.py --setup 安装：提交后自动推送，不再触发 commit。\n"
            f'"{py}" "{script}" --auto >> "{LOG.as_posix()}" 2>&1 &\n',
            encoding="utf-8", newline="\n")
        try:
            hook.chmod(0o755)
        except Exception:
            pass
        print(f"[setup] 已安装提交后自动推送 hook：{hook.relative_to(ROOT)}")
    else:
        print("[setup] 按要求不安装自动 hook，需手动跑同步命令")
    return 0


def store_token(token: str | None = None) -> int:
    """把 PAT 交给 Git Credential Manager 持久化，之后所有推送免密。

    GCM 的浏览器弹窗 / 设备码在某些无界面环境里不吐码也弹不出窗口，
    这时用网页生成一次 classic PAT 最稳。token 只经 stdin 传给
    `git credential approve`，不写项目文件、不入库、不进日志。
    """
    tok = (token or os.environ.get("GITHUB_TOKEN")
           or os.environ.get("GH_TOKEN") or "").strip()
    if not tok:
        print("[store-token] 没拿到 token：设 GITHUB_TOKEN 环境变量或传 --token")
        return 1
    payload = (f"protocol=https\nhost=github.com\n"
               f"username=x-access-token\npassword={tok}\n\n")
    r = subprocess.run(["git", "-C", str(ROOT), "credential", "approve"],
                       input=payload, text=True, capture_output=True)
    if r.returncode != 0:
        print("[store-token] 存入失败：", r.stderr.strip()[:200])
        return 1
    print("[store-token] 凭据已存入 Windows 凭据管理器（GCM），之后推送不再需要 token")
    return 0


def status() -> int:
    url = git("remote", "get-url", "origin")
    branch = git("branch", "--show-current").stdout.strip()
    print(f"origin  : {url.stdout.strip() if url.returncode == 0 else '(未配置)'}")
    print(f"branch  : {branch}")
    print(f"HEAD    : {git('rev-parse', '--short', 'HEAD').stdout.strip()}")
    if url.returncode == 0:
        r = git("rev-list", "--left-right", "--count", f"origin/{branch}...HEAD")
        print(f"远端/本地差异 (落后 领先): {r.stdout.strip() or '(远端尚无该分支)'}")
    st = git("status", "--short")
    lines = [x for x in (st.stdout or "").splitlines() if x.strip()]
    print(f"未提交改动: {len(lines)} 项")
    for x in lines[:10]:
        print("  ", x)
    return 0


def sync(message: str | None, auto: bool, dry_run: bool) -> int:
    branch = git("branch", "--show-current").stdout.strip() or "master"
    if not auto:
        if dry_run:
            print("[dry-run] 将执行：git add -A / commit / push")
            return 0
        a = git("add", "-A")
        if a.returncode != 0:
            print("[error] git add 失败：", a.stderr.strip())
            return 1
        st = git("status", "--porcelain")
        if st.stdout.strip():
            msg = message or f"sync: 本地更新 {datetime.now():%Y-%m-%d %H:%M}"
            c = git("commit", "-m", msg)
            if c.returncode != 0:
                print("[error] commit 失败：", c.stderr.strip())
                return 1
            print(f"[sync] 已提交：{c.stdout.strip().splitlines()[0]}")
        else:
            print("[sync] 无改动，直接推送")
    else:
        print("[auto] 只推送，不提交（避免 hook 递归）")

    url = git("remote", "get-url", "origin")
    if url.returncode != 0:
        msg = "未配置 origin，跳过推送"
        log(msg)
        print(f"[skip] {msg}；先跑：python scripts/github_sync.py --setup <仓库地址>")
        return 0 if auto else 1

    cfg = _token_args()
    p = git("push", "-u", "origin", f"HEAD:{branch}", extra_config=cfg)
    if p.returncode == 0:
        msg = f"推送成功 → {url.stdout.strip()} ({branch})"
        log(msg)
        print(f"[sync] {msg}")
        return 0
    err = (p.stderr or p.stdout or "").strip()
    log(f"推送失败：{err[-600:]}")
    print("[sync] 推送失败：", err[-600:])
    print("        若提示认证失败：设 GITHUB_TOKEN 后重试，或首次推送时完成 GCM 弹窗授权")
    return 0 if auto else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="AutoResearch ↔ GitHub 同步")
    ap.add_argument("--setup", metavar="URL", help="配置 origin 并安装自动推送 hook")
    ap.add_argument("--no-hook", action="store_true", help="--setup 时不装 hook")
    ap.add_argument("--status", action="store_true", help="查看远端与本地差异")
    ap.add_argument("--auto", action="store_true", help="post-commit hook 内部调用")
    ap.add_argument("--store-token", action="store_true",
                    help="把 GITHUB_TOKEN / --token 存进系统凭据管理器，之后免密推送")
    ap.add_argument("--token", default="", help="配合 --store-token；也可只用环境变量")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-m", "--message", default="", help="提交信息")
    args = ap.parse_args()
    if args.setup:
        return setup(args.setup, install_hook=not args.no_hook)
    if args.store_token:
        return store_token(args.token)
    if args.status:
        return status()
    return sync(args.message or None, args.auto, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
