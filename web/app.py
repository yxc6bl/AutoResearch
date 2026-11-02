"""AutoResearch 网页界面（零外部依赖，仅用标准库）。

启动：  python web/app.py
访问：  http://127.0.0.1:5000

功能：
- 填研究主题 / 选数据集 / 选模式（plan 单次 / iterate 多轮），点运行，后台跑流水线；
- 实时显示运行日志；
- 运行结束展示：论文全文、六维评分卡片、文献候选、实验指标、图表；
- "打开已有结果目录"：直接加载本地已生成的产物（如 out_eval3/round_1），无需重跑（避开 API 限流）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # AutoResearch/
sys.path.insert(0, str(ROOT))

from web.render import collect_result, verdict_to_html  # noqa: E402

# 发布到云端时由启动环境注入；默认 0.0.0.0 允许外部访问，本地调试可设 WEB_HOST=127.0.0.1
HOST = os.environ.get("WEB_HOST", "0.0.0.0")
PORT = int(os.environ.get("WEB_PORT", "5000"))

# 内存任务表：task_id -> {proc, log(list), status, out_dir, error}
TASKS: dict[str, dict] = {}
_lock = threading.Lock()
_task_seq = 0

# 可选内置数据集（sklearn）
DATASETS = ["breast_cancer", "iris", "wine", "digits", "load_diabetes", "fetch_california_housing"]


def _next_id() -> str:
    global _task_seq
    with _lock:
        _task_seq += 1
        return f"task_{_task_seq:04d}"


def run_pipeline(task_id: str, mode: str, topic: str, dataset: str, rounds: int,
                 out_dir: Path, api_key: str | None = None):
    """在子线程里跑流水线，把 stdout/stderr 累积到 TASKS[task_id]['log']。

    api_key：用户在网页填入的 Key，仅注入本次子进程环境变量，不落盘、不写日志。
    优先级高于服务端 .env；留空则回退到服务端 .env 配置。
    """
    cmd = [sys.executable, "auto_research.py"]
    if mode == "iterate":
        cmd += ["iterate", "--topic", topic, "--rounds", str(rounds), "--out", str(out_dir)]
        if dataset:
            cmd += ["--dataset", dataset]
    else:  # plan
        cmd += ["plan", "--topic", topic, "--out", str(out_dir)]
        if dataset:
            cmd += ["--dataset", dataset]

    # 仅注入本次运行所需环境变量；auto_research 的 load_dotenv 不会覆盖已存在的 Key
    env = os.environ.copy()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
        env["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL", "https://apiai.sztu.edu.cn/v1")
        env["OPENAI_MODEL"] = env.get("OPENAI_MODEL", "deepseek-v4-pro")

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", bufsize=1, env=env,
        )
        with _lock:
            TASKS[task_id]["proc"] = proc
        for line in proc.stdout:
            with _lock:
                TASKS[task_id]["log"].append(line.rstrip("\n"))
        proc.wait()
        with _lock:
            TASKS[task_id]["status"] = "done" if proc.returncode == 0 else "failed"
            TASKS[task_id]["returncode"] = proc.returncode
    except Exception as e:  # noqa: BLE001
        with _lock:
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["error"] = str(e)
            TASKS[task_id]["log"].append(f"[server] 启动失败：{e}")


def build_result_payload(out_dir: Path) -> dict:
    res = collect_result(str(out_dir))
    res["verdict_html"] = verdict_to_html(res["verdict"]) if res["verdict"] else ""
    # 把 plot 绝对路径转成可被 /api/file 解析的相对工程根路径
    res["plots_rel"] = [Path(p).relative_to(ROOT).as_posix() for p in res["plots"]]
    res["plots"] = []  # 前端改用 plots_rel
    return res


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/datasets":
            self._json({"datasets": DATASETS})
            return

        if path == "/api/task" and "id" in qs:
            tid = qs["id"][0]
            with _lock:
                t = TASKS.get(tid, {})
                log = t.get("log", [])[-200:]
                payload = {
                    "status": t.get("status"),
                    "log": log,
                    "out_dir": t.get("out_dir"),
                    "error": t.get("error"),
                }
            self._json(payload)
            return

        if path == "/api/result" and "id" in qs:
            tid = qs["id"][0]
            with _lock:
                out_dir = TASKS.get(tid, {}).get("out_dir")
            if not out_dir:
                self._json({"found": False, "error": "任务不存在"}, 404)
                return
            try:
                self._json(build_result_payload(Path(out_dir)))
            except Exception as e:  # noqa: BLE001
                self._json({"found": False, "error": str(e)}, 500)
            return

        if path == "/api/open" and "dir" in qs:
            d = qs["dir"][0]
            p = Path(d)
            if not p.is_absolute():
                p = ROOT / p
            if not p.exists():
                self._json({"found": False, "error": f"目录不存在：{p}"}, 404)
                return
            try:
                self._json(build_result_payload(p))
            except Exception as e:  # noqa: BLE001
                self._json({"found": False, "error": str(e)}, 500)
            return

        if path == "/api/file":
            rel = qs.get("path", [""])[0]
            fp = (ROOT / rel).resolve()
            if not str(fp).startswith(str(ROOT.resolve())):
                self._send(403, b"forbidden")
                return
            if not fp.exists():
                self._send(404, b"not found")
                return
            ctype = "image/png" if fp.suffix == ".png" else "application/octet-stream"
            self._send(200, fp.read_bytes(), ctype)
            return

        self._send(404, b"not found")

    def do_POST(self):
        if self.path != "/api/run":
            self._send(404, b"not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = {}

        mode = data.get("mode", "plan")
        topic = (data.get("topic") or "").strip()
        dataset = (data.get("dataset") or "").strip()
        rounds = int(data.get("rounds", 3))
        api_key = (data.get("api_key") or "").strip() or None

        if mode == "open":  # 打开已有目录，不跑流水线
            d = (data.get("dir") or "").strip()
            p = Path(d)
            if not p.is_absolute():
                p = ROOT / p
            tid = _next_id()
            with _lock:
                TASKS[tid] = {"status": "done", "log": [f"[open] 加载 {p}"],
                              "out_dir": str(p), "proc": None}
            self._json({"task_id": tid, "open": True})
            return

        if not topic:
            self._json({"error": "主题不能为空"}, 400)
            return

        tid = _next_id()
        out_dir = ROOT / "out_web" / tid
        out_dir.mkdir(parents=True, exist_ok=True)
        with _lock:
            TASKS[tid] = {"status": "running", "log": [f"[server] 启动流水线：mode={mode} topic={topic}"],
                          "out_dir": str(out_dir), "proc": None}
        threading.Thread(
            target=run_pipeline, args=(tid, mode, topic, dataset, rounds, out_dir, api_key),
            daemon=True,
        ).start()
        self._json({"task_id": tid})

    def log_message(self, *args):  # 静默默认访问日志
        pass


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"AutoResearch 网页界面已启动： http://{HOST}:{PORT}")
    print(f"工程目录： {ROOT}")
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
