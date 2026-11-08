#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成可直接交给别人的共享包（不依赖 GitHub、不依赖授权）。

打包原则：只放「可复现的工程」，不放半成品与密钥。

  放：代码、公开数据集、全部文档、三篇已过 G04 的论文（含 Word 与图）、
      一键启动脚本、.env.example、使用说明。
  不放：.git/、out_*/、.env、*.log、*.pyc、__pycache__/、corpus/、
      memory/data/、demo_results/、dist/、任何 .zip。

用法：  python scripts/make_share_package.py
输出：  dist/AutoResearch_share_<日期>.zip
"""
from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PREFIX = "AutoResearch"

SKIP_DIRS = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".workbuddy",
    "corpus", "demo_results", "dist", "venv", ".venv", "env",
    "node_modules", ".idea", ".vscode",
}
# memory/store.py 被四个子代理 import，必须进包；只排掉本地台账与缓存数据
SKIP_REL_PARTS = {("memory", "data")}
SKIP_REL_FILES = {"memory/feedback.jsonl", "memory/lit_usage.json"}
SKIP_FILE_SUFFIX = {".log", ".pyc", ".pyo", ".zip", ".bak", ".tmp"}
SKIP_FILE_NAMES = {".env", ".DS_Store", "Thumbs.db"}

# 三篇示例成果：三种「菜单外」后端各一篇，都带真实评审结论与 Word
EXAMPLES = {
    "association_paper": "out_assoc",
    "graph_paper": "out_graph_cli",
    "image_paper": "out_image_cli",
}
EXAMPLE_FILES = [
    "task_spec.json", "01_literature.json", "02_hypotheses.json",
    "03_experiment.json", "07_gate_report.json", "00_trace.jsonl",
    "06_paper/paper_full.md", "06_paper/paper_full.docx",
    "06_paper/07_eval_verdict.json", "06_paper/DELIVERY_STATUS.json",
    "06_paper/WRITE_STATUS.json", "06_paper/REVIEW_ROUNDS.json",
]

START_WEB = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键启动本地网站：python start_web.py  （会自动打开浏览器）"""
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
host = os.environ.get("WEB_HOST", "127.0.0.1")
port = int(os.environ.get("WEB_PORT", "5000"))
env = os.environ.copy()
env["WEB_HOST"] = host
env["WEB_PORT"] = str(port)
url = f"http://{host}:{port}"
print(f"网站地址：{url}")
print("关掉这个窗口即可停止服务。")
try:
    webbrowser.open(url)
except Exception:
    pass
sys.exit(subprocess.call(
    [sys.executable, "-u", str(ROOT / "web" / "app.py")], cwd=str(ROOT), env=env))
'''

README = '''# 使用说明

这是一个可以离线跑的论文生成工程。不需要 GitHub，也不需要账号。

## 最快看到效果

1. 装依赖：`python -m pip install -r requirements.txt`
2. 启动网站：`python start_web.py`（会自动打开浏览器）
3. 不填 API Key 也能看成果：在页面「打开已有结果」里填
   `examples/image_paper`（或 `examples/graph_paper`、`examples/association_paper`），
   直接看到论文全文、评分卡和图表。

## 自己跑一篇新论文

在网页填 API Key（必填，只用于本次运行、不保存）→ 填研究题目 → 选数据集 → 点运行。
可选数据集里 `market_basket` 是关联规则、`cora_graph` 是图节点分类、
`mnist_image` 是图像分类。

不想开网页也能用命令行：

    python auto_research.py backends                 # 看七条已接线后端
    python auto_research.py backends --probe         # 真导入检查依赖
    python auto_research.py plan --topic "基于卷积神经网络的手写数字图像分类研究" --catalog-key mnist --out out_x
    python auto_research.py gates --out out_x        # 看能不能交付

API Key 配置模板见 `.env.example`；不要把自己的 Key 写进任何会被分享的文件。

## 包里有什么

- 完整代码、公开数据集（Cora 引文网络、MNIST、UCI/NASA 表格数据等）、全部文档；
- `examples/` 下三篇已通过 G04 零编造硬门的可交付论文，含 Markdown、Word、
  评审结论和图表；
- `docs/CAPABILITY.md` 是能力现状盘点，`docs/ARCHITECTURE.md` 是架构论证，
  `docs/BACKENDS.md` 是补新领域后端的操作手册。

不包含 `.git/`、任何密钥、运行日志与 `out_*/` 半成品。
'''


def _should_skip(path: Path, is_dir: bool) -> bool:
    if is_dir:
        return path.name in SKIP_DIRS or path.name.startswith("out_")
    if path.name in SKIP_FILE_NAMES:
        return True
    if path.suffix.lower() in SKIP_FILE_SUFFIX:
        return True
    # .env.* 里只保留模板
    if path.name.startswith(".env.") and path.name != ".env.example":
        return True
    return False


def build() -> Path:
    DIST.mkdir(exist_ok=True)
    out = DIST / f"AutoResearch_share_{datetime.now():%Y%m%d}.zip"
    added = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(ROOT.rglob("*")):
            rel = path.relative_to(ROOT)
            if any(p in SKIP_DIRS for p in rel.parts):
                continue
            if len(rel.parts) >= 2 and (rel.parts[0], rel.parts[1]) in SKIP_REL_PARTS:
                continue
            if rel.as_posix() in SKIP_REL_FILES:
                continue
            if rel.parts and rel.parts[0].startswith("out_"):
                continue
            if _should_skip(path, path.is_dir()):
                continue
            if path.is_dir():
                continue
            z.write(path, f"{PREFIX}/{rel.as_posix()}")
            added += 1

        # 示例成果
        for name, src in EXAMPLES.items():
            base = ROOT / src
            n = 0
            for f in EXAMPLE_FILES:
                p = base / f
                if p.exists():
                    z.write(p, f"{PREFIX}/examples/{name}/{f}")
                    n += 1
                    added += 1
            for p in sorted((base / "plots").glob("*.png")):
                z.write(p, f"{PREFIX}/examples/{name}/plots/{p.name}")
                n += 1
                added += 1
            print(f"  examples/{name}: {n} 个文件")

        z.writestr(f"{PREFIX}/start_web.py", START_WEB)
        z.writestr(f"{PREFIX}/使用说明.md", README)
        added += 2

    print(f"已生成：{out}")
    print(f"共 {added} 个文件，{out.stat().st_size / 1024 / 1024:.1f} MB")
    return out


if __name__ == "__main__":
    build()
