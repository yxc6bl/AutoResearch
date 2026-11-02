"""记忆库（MemoryStore）—— 所有子代理共用的共享状态「黑板」。

这是「接口契约」的参考实现（本地 JSON 版）。组员后续可替换为
数据库 / 向量库等后端，只要保持下方 get/put/query/list 四个方法的
签名与语义不变，各子代理无需改动（演进只做加法，不改名、不删字段）。

命名空间约定（详见 docs/FRAMEWORK.md）：
    literature/   文献记忆（lit_memory）—— 记住抓过什么文献
    user/         跨项目长期用户画像（user_profile）—— 蒸馏后的「这个人」
    project/      当前项目细节记忆（project_memory）
    feedback/     用户明确反馈记忆（feedback_memory）

存储：每个命名空间一个 JSON 文件，位于 <root>/memory/data/<ns>.json。
该目录已在 .gitignore 排除（运行时数据，不入库）。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

DEFAULT_ROOT = Path(__file__).resolve().parent
# 命名空间白名单：加新命名空间只在此追加（加法演进）
NAMESPACES = ("literature", "user", "project", "feedback")


class MemoryStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = (root or DEFAULT_ROOT) / "data"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ── 内部工具 ──
    def _path(self, ns: str) -> Path:
        if ns not in NAMESPACES:
            raise ValueError(f"未知命名空间 {ns!r}，应为 {NAMESPACES}")
        return self.root / f"{ns}.json"

    def _read(self, ns: str) -> Any:
        p = self._path(ns)
        if not p.exists():
            return {"papers": []} if ns == "literature" else {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"papers": []} if ns == "literature" else {}

    def _write(self, ns: str, data: Any) -> None:
        p = self._path(ns)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 契约接口（稳定，演进只加不改） ──
    def get(self, ns: str, key: str) -> Optional[Any]:
        """按 命名空间 + 主键 取一条记录；不存在返回 None。
        literature 以 paper id 为主键；其余以各自顶层 key 为主键。"""
        with self._lock:
            data = self._read(ns)
            if ns == "literature":
                for p in data.get("papers", []):
                    if p.get("id") == key:
                        return p
                return None
            return data.get(key)

    def put(self, ns: str, key: str, record: Any) -> None:
        """写入 / 覆盖一条记录。
        literature 以 record['id']（缺省时取 key）为主键去重合并 cited_in；
        其余命名空间以 key 为主键覆盖。"""
        with self._lock:
            data = self._read(ns)
            if ns == "literature":
                papers = data.setdefault("papers", [])
                rid = record.get("id") or key
                record = dict(record)
                record["id"] = rid
                for i, p in enumerate(papers):
                    if p.get("id") == rid:
                        old_cited = set(p.get("cited_in", []))
                        new_cited = set(record.get("cited_in", []))
                        merged = dict(p)
                        merged.update(record)
                        merged["cited_in"] = sorted(old_cited | new_cited)
                        papers[i] = merged
                        self._write(ns, data)
                        return
                papers.append(record)
            else:
                data[key] = record
            self._write(ns, data)

    def query(self, ns: str, filt: Optional[dict] = None) -> list[Any]:
        """按字段过滤返回记录列表；filt 为空返回全部。"""
        with self._lock:
            data = self._read(ns)
            items = data.get("papers", []) if ns == "literature" else list(data.values())
            if not filt:
                return items
            return [it for it in items if all(it.get(k) == v for k, v in filt.items())]

    def list(self, ns: str) -> list[str]:
        """返回命名空间下所有主键。"""
        with self._lock:
            data = self._read(ns)
            if ns == "literature":
                return [p.get("id") for p in data.get("papers", [])]
            return list(data.keys())

    def delete(self, ns: str, key: str) -> bool:
        """删除一条记录：literature 按 paper id 删，其余按顶层 key 删。
        返回是否确实删掉了内容。"""
        with self._lock:
            data = self._read(ns)
            if ns == "literature":
                before = data.get("papers", [])
                after = [p for p in before if p.get("id") != key]
                if len(after) == len(before):
                    return False
                data["papers"] = after
            else:
                if key not in data:
                    return False
                del data[key]
            self._write(ns, data)
            return True


# 便捷单例：各子代理直接 `from memory.store import store`
store = MemoryStore()
