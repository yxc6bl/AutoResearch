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
import re
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # AutoResearch/
sys.path.insert(0, str(ROOT))

from web.render import collect_result, find_paper_dir, verdict_to_html  # noqa: E402

# 发布到云端时由启动环境注入；默认 0.0.0.0 允许外部访问，本地调试可设 WEB_HOST=127.0.0.1
HOST = os.environ.get("WEB_HOST", "0.0.0.0")
# 发布沙箱注入标准 PORT 环境变量；本地调试可用 WEB_PORT，再回退 5000
PORT = int(os.environ.get("PORT") or os.environ.get("WEB_PORT") or "5000")

# 内存任务表：task_id -> {proc, log(list), status, out_dir, error}
TASKS: dict[str, dict] = {}
_lock = threading.Lock()
_task_seq = 0

# 可选内置数据集（sklearn）
DATASETS = ["breast_cancer", "iris", "wine", "digits", "load_diabetes",
            "fetch_california_housing", "market_basket", "cora_graph", "mnist_image"]


def _next_id() -> str:
    global _task_seq
    with _lock:
        _task_seq += 1
        return f"task_{_task_seq:04d}"


def run_pipeline(task_id: str, mode: str, topic: str, dataset: str, rounds: int,
                 out_dir: Path, api_key: str | None = None,
                 data_path: str | None = None, data_url: str | None = None,
                 target: str = "", sep: str = "", paper_type: str = "method_comparison",
                 type_material: dict | None = None):
    """在子线程里跑流水线，把 stdout/stderr 累积到 TASKS[task_id]['log']。

    api_key：用户在网页填入的 Key，仅注入本次子进程环境变量，不落盘、不写日志。
    优先级高于服务端 .env；留空则回退到服务端 .env 配置。
    data_path / data_url：用户自带 CSV 或公开数据集网址（二选一，覆盖内置 dataset）。
    """
    # -u：子进程无缓冲输出。否则 stdout 接管道时是块缓冲，网页上的「实时日志」
    # 要等缓冲区写满或进程退出才刷新，用户会误以为卡死。
    cmd = [sys.executable, "-u", "auto_research.py"]
    if mode == "iterate":
        # iterate 仅用于方法对比型扩宽数据集/方法，默认 method_comparison
        cmd += ["iterate", "--topic", topic, "--rounds", str(rounds), "--out", str(out_dir)]
    else:  # plan
        cmd += ["plan", "--topic", topic, "--out", str(out_dir)]
        if paper_type and paper_type != "method_comparison":
            cmd += ["--paper-type", paper_type]
        # 论文类型素材（system_spec / case_spec / ...）：写盘后由 --spec-extra 挂载
        if type_material:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                _extra = out_dir / "_spec_extra.json"
                _extra.write_text(
                    json.dumps({"material": type_material}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                cmd += ["--spec-extra", str(_extra)]
            except Exception as _e:  # noqa: BLE001
                print(f"[server] 类型素材写入失败，忽略：{_e}")

    # 数据集来源优先级：网址 > 上传文件 > 内置/自动
    if data_url:
        cmd += ["--url", data_url]
        if target:
            cmd += ["--target", target]
        if sep:
            cmd += ["--sep", sep]
    elif data_path:
        cmd += ["--data", str(data_path)]
        if target:
            cmd += ["--target", target]
        if sep:
            cmd += ["--sep", sep]
    elif dataset:
        if dataset == "market_basket":
            cmd += ["--data", str(ROOT / "data" / "market_basket.csv")]
        elif dataset == "fetch_california_housing":
            # sklearn 的 fetch_* 需联网下载（实测 403），改绑本地 CSV 快照。
            cmd += ["--catalog-key", "california_housing"]
        elif dataset == "cora_graph":
            # 图数据要 nodes + edges 两个文件，不能走 --data/--dataset 的单表入口。
            cmd += ["--catalog-key", "cora"]
        elif dataset == "mnist_image":
            # 图像是 IDX 张量目录，同样走目录绑定而不是单表入口。
            cmd += ["--catalog-key", "mnist"]
        else:
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


def _recent_runs(limit: int = 9) -> list[dict]:
    """扫描工程根下的产物的最新动态，供网页「最近产物」一键打开。"""
    cands: list[tuple[float, Path, Path]] = []
    roots = [p for p in ROOT.glob("out_*") if p.is_dir()]
    demo = ROOT / "demo_results"
    if demo.is_dir():
        roots += [p for p in demo.glob("*") if p.is_dir()]
    for d in roots:
        try:
            pd = find_paper_dir(d)
            if not pd:
                continue
            cands.append(((pd / "paper_full.md").stat().st_mtime, d, pd))
        except Exception:  # noqa: BLE001
            continue
    cands.sort(key=lambda x: x[0], reverse=True)

    rows = []
    for mtime, d, pd in cands[: max(1, limit)]:
        try:
            md = (pd / "paper_full.md").read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        title = ""
        for ln in md.split("\n"):
            if ln.startswith("# "):
                title = ln[2:].strip()
                break
        v = {}
        try:
            v = json.loads((pd / "07_eval_verdict.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        wst = {}
        try:
            wst = json.loads((pd / "WRITE_STATUS.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        rows.append({
            "dir": d.relative_to(ROOT).as_posix(),
            "name": d.name,
            "title": title or d.name,
            "paper_type": wst.get("paper_type") or "",
            "chars": len(re.findall(r"[\u4e00-\u9fff]", md)),
            "score": v.get("overall"),
            "passed": v.get("passed"),
            "gate": v.get("gate") or "",
            "mtime": mtime,
        })
    return rows


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8",
              extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _safe_dir(self, raw: str):
        """只允许访问工程根内的产物目录，避免任意路径读取。"""
        d = Path(raw)
        if not d.is_absolute():
            d = ROOT / d
        try:
            d = d.resolve()
            root = ROOT.resolve()
            if d != root and root not in d.parents:
                return None
        except Exception:
            return None
        return d if d.exists() else None

    def _get_lit_choices(self, raw_dir: str):
        d = self._safe_dir(raw_dir)
        if d is None:
            self._json({"found": False, "error": "目录不存在或不在工程内"}, 404)
            return
        try:
            from engine import selection_ledger as SL
            from engine import lit_plan as LP
            from paper.lit_material import normalize_gist as _ng
            lp = d / "01_literature.json"
            papers = json.loads(lp.read_text(encoding="utf-8")).get("papers") or [] if lp.exists() else []
            prefs = SL.load_prefs()
            drops = {x["key"] for x in prefs.get("paper_decisions", []) if x.get("action") == "drop"}
            keeps = {x["key"] for x in prefs.get("paper_decisions", []) if x.get("action") == "keep"}
            rows = []
            for i, p in enumerate(papers, 1):
                key = str(p.get("id") or p.get("title") or "")
                state = "drop" if SL._paper_hits(drops, p) else (
                    "keep" if SL._paper_hits(keeps, p) else "ai")
                rows.append({"num": i, "key": key, "title": p.get("title") or "",
                             "year": p.get("year"), "venue": p.get("venue"),
                             "gist": _ng(p.get("gist_cn")),
                             "human": state})
            plan = LP.load(d) or {}
            self._json({
                "found": True, "dir": str(d), "papers": rows,
                "plan_available": bool(plan),
                "plan": {"sources": (plan.get("strategy") or {}).get("sources") or [],
                         "query_groups": (plan.get("strategy") or {}).get("query_groups") or [],
                         "applied_feedback": plan.get("applied_feedback") or [],
                         "outcome": plan.get("outcome") or {}},
                "n_rules": len(prefs.get("rules") or []),
                "n_decisions": len(prefs.get("paper_decisions") or []),
            })
        except Exception as e:  # noqa: BLE001
            self._json({"found": False, "error": str(e)}, 500)

    def _post_lit_choices(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = {}
        d = self._safe_dir(str(data.get("dir") or ""))
        if d is None:
            self._json({"ok": False, "error": "目录不存在或不在工程内"}, 404)
            return
        try:
            from engine import selection_ledger as SL
            res = SL.apply_human_choices(
                d, data.get("keep") or [], data.get("drop") or [],
                str(data.get("note") or "")[:200])
            res["ok"] = True
            res["hint"] = "已写入记忆：下次抓取会按这些选择剔除/优先保留；本次选材台账已标注 by=human。"
            self._json(res)
        except Exception as e:  # noqa: BLE001
            self._json({"ok": False, "error": str(e)}, 500)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8",
                       {"Cache-Control": "no-store"})
            return

        if path == "/favicon.ico":
            self._send(204, b"")
            return

        if path == "/api/health":
            self._json({"ok": True, "root": str(ROOT), "tasks": len(TASKS)})
            return

        if path == "/api/recent":
            try:
                n = int(qs.get("limit", ["9"])[0])
            except Exception:  # noqa: BLE001
                n = 9
            self._json({"runs": _recent_runs(max(1, min(30, n)))})
            return

        if path == "/api/lit-choices" and "dir" in qs:
            self._get_lit_choices(qs["dir"][0])
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
            d = self._safe_dir(qs["dir"][0])
            if d is None:
                self._json({"found": False, "error": "目录不存在或不在工程内"}, 404)
                return
            try:
                self._json(build_result_payload(d))
            except Exception as e:  # noqa: BLE001
                self._json({"found": False, "error": str(e)}, 500)
            return

        if path == "/api/file":
            rel = qs.get("path", [""])[0]
            fp = (ROOT / rel).resolve()
            root = ROOT.resolve()
            if fp != root and root not in fp.parents:
                self._send(403, b"forbidden")
                return
            if not fp.is_file():
                self._send(404, b"not found")
                return
            ctype = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".svg": "image/svg+xml", ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".md": "text/markdown; charset=utf-8", ".json": "application/json; charset=utf-8",
                ".csv": "text/csv; charset=utf-8",
            }.get(fp.suffix.lower(), "application/octet-stream")
            extra = {"Cache-Control": "no-store"}
            if qs.get("download", [""])[0] in ("1", "true", "yes"):
                fn = urllib.parse.quote(fp.name)
                extra["Content-Disposition"] = (
                    f"attachment; filename=\"{fp.name.encode('ascii', 'ignore').decode() or 'download'}\"; "
                    f"filename*=UTF-8''{fn}")
            self._send(200, fp.read_bytes(), ctype, extra)
            return

        self._send(404, b"not found")

    def do_HEAD(self):
        """让 curl -I / 浏览器预检能拿到状态码与类型，不必先下整个文件。"""
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/file":
            rel = urllib.parse.parse_qs(parsed.query).get("path", [""])[0]
            fp = (ROOT / rel).resolve()
            root = ROOT.resolve()
            if fp != root and root not in fp.parents:
                self.send_response(403)
            elif not fp.is_file():
                self.send_response(404)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(fp.stat().st_size))
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/lit-choices":
            self._post_lit_choices()
            return
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
        paper_type = (data.get("paper_type") or "method_comparison").strip()
        # 合法类型以论文类型注册表为准（新增类型不必改这里）
        try:
            from paper.thesis_template import paper_type_registry
            _valid_pt = set(paper_type_registry().keys())
        except Exception:  # noqa: BLE001
            _valid_pt = {"method_comparison", "survey", "system_design"}
        if paper_type not in _valid_pt:
            paper_type = "method_comparison"
        # 论文类型素材：前端传 JSON 对象（或 JSON 字符串），原样转交 --spec-extra
        type_material = data.get("type_material")
        if isinstance(type_material, str):
            try:
                type_material = json.loads(type_material) if type_material.strip() else None
            except Exception:  # noqa: BLE001
                type_material = None
        if not isinstance(type_material, dict) or not type_material:
            type_material = None
        rounds = int(data.get("rounds", 3))
        api_key = (data.get("api_key") or "").strip() or None
        target = (data.get("target") or "").strip()
        sep = (data.get("sep") or "").strip()
        data_url = (data.get("data_url") or "").strip() or None
        data_b64 = (data.get("data_b64") or "").strip()
        data_filename = (data.get("data_filename") or "upload.csv").strip()

        if mode == "open":  # 打开已有目录，不跑流水线
            d = self._safe_dir((data.get("dir") or "").strip())
            if d is None:
                self._json({"error": "目录不存在或不在工程内"}, 404)
                return
            tid = _next_id()
            with _lock:
                TASKS[tid] = {"status": "done", "log": [f"[open] 加载 {d}"],
                              "out_dir": str(d), "proc": None}
            self._json({"task_id": tid, "open": True})
            return

        if not topic:
            self._json({"error": "主题不能为空"}, 400)
            return

        # 处理上传文件（base64 解码落盘）
        data_path = None
        if data_b64:
            import base64
            try:
                # 去掉 data:xxx;base64, 前缀
                if "," in data_b64:
                    data_b64 = data_b64.split(",", 1)[1]
                blob = base64.b64decode(data_b64)
                suffix = ".xlsx" if data_filename.lower().endswith((".xlsx", ".xls")) else ".csv"
                out_dir = ROOT / "out_web" / "_uploads"
                out_dir.mkdir(parents=True, exist_ok=True)
                fpath = out_dir / f"upload_{_next_id()}{suffix}"
                fpath.write_bytes(blob)
                data_path = str(fpath)
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"文件解码失败：{e}"}, 400)
                return

        tid = _next_id()
        out_dir = ROOT / "out_web" / tid
        out_dir.mkdir(parents=True, exist_ok=True)
        log_head = f"[server] 启动流水线：mode={mode} topic={topic} paper_type={paper_type}"
        if type_material:
            log_head += f" | 类型素材字段={len(type_material)}"
        if data_url:
            log_head += f" | 数据集网址={data_url}"
        elif data_path:
            log_head += f" | 上传文件={data_path}"
        else:
            log_head += f" | 数据集={dataset or '自动'}"
        with _lock:
            TASKS[tid] = {"status": "running", "log": [log_head],
                          "out_dir": str(out_dir), "proc": None}
            # 只保留最近 60 个已结束任务，防止长驻进程内存无限增长
            if len(TASKS) > 60:
                for k in sorted(TASKS)[: len(TASKS) - 60]:
                    if TASKS.get(k, {}).get("status") != "running":
                        TASKS.pop(k, None)
        threading.Thread(
            target=run_pipeline,
            args=(tid, mode, topic, dataset, rounds, out_dir, api_key, data_path, data_url,
                  target, sep, paper_type, type_material),
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
