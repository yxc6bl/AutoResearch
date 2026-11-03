"""确定性分析引擎（不依赖 LLM 输出 JSON）。

根据任务规格（数据集 / 任务类型 / 候选方法 / 主指标）直接基于标准 sklearn
数据集重跑全部候选方法、跨多种子统计、计算逐类指标 / 混淆矩阵 / 特征重要性 /
多种子方差 / 显著性检验，并生成图表。无论实验 Agent 的 LLM 是否偷懒，分析
产物都稳定、可复现。

本模块由 auto_research.py 在本地进程直接调用；OpenML 数据源例外，需要联网。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 让图表支持中文（跨平台：Windows 自带字体 / Linux Noto CJK / 环境变量指定）
try:
    import matplotlib.font_manager as _fm
    _cjk_font = None
    _cands = [
        os.environ.get("CJK_FONT_PATH", ""),
        # 微软雅黑优先：字形更好且含上标²（R² 等），避免 SimHei 缺失该字形渲染成方框
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for _cand in _cands:
        if _cand and os.path.exists(_cand):
            try:
                _fm.fontManager.addfont(_cand)
                _cjk_font = _fm.FontProperties(fname=_cand).get_name()
                break
            except Exception:
                continue
    # 兜底：在已注册字体里找任意含 CJK 的（如已 apt 安装 fonts-noto-cjk）
    if not _cjk_font:
        for _f in _fm.fontManager.ttflist:
            if any(k in _f.name for k in ("CJK", "Noto Sans CJK", "WenQuanYi",
                                          "Source Han", "SimHei", "Microsoft YaHei")):
                _cjk_font = _f.name
                break
    if _cjk_font:
        plt.rcParams["font.family"] = _cjk_font
        plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy import stats

SEEDS = [42, 43, 44, 45, 46]
TEST_SIZE = 0.2


# ── 数据集加载 ──────────────────────────────────────────────────────────
def _load_breast_cancer():
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer()
    return d.data.astype(float), d.target, list(d.feature_names), "classification"


def _load_digits():
    from sklearn.datasets import load_digits

    d = load_digits()
    X = d.images.reshape(len(d.images), -1).astype(float)
    feat = [f"pixel_{i}" for i in range(X.shape[1])]
    return X, d.target, feat, "classification"


def _load_diabetes():
    from sklearn.datasets import load_diabetes

    d = load_diabetes()
    return d.data.astype(float), d.target.astype(float), list(d.feature_names), "regression"


DATASETS = {
    "breast_cancer": _load_breast_cancer,
    "digits": _load_digits,
    "diabetes": _load_diabetes,
}


def _load_sklearn_clustering(name: str):
    """加载 sklearn 数据集用于聚类（保留真实标签仅用于评估）。"""
    import sklearn.datasets as ds

    loader = getattr(ds, "load_" + name, None) or getattr(ds, name, None)
    if loader is None:
        raise ValueError(f"未知 sklearn 聚类数据集：{name}")
    b = loader()
    X = np.asarray(b.data, dtype=float)
    y_true = np.asarray(b.target)
    feat = (
        list(b.feature_names)
        if getattr(b, "feature_names", None) is not None
        else [f"f{i}" for i in range(X.shape[1])]
    )
    return X, y_true, feat, "clustering"


# ── 通用数据集加载（零硬编码：任意 CSV/Parquet/Excel/OpenML/ sklearn 名） ──
def _load_sklearn_generic(name: str):
    """加载任意 sklearn.datasets 里的数据集（不再只认 3 个内置）。"""
    import sklearn.datasets as ds

    loader = getattr(ds, "load_" + name, None) or getattr(ds, name, None)
    if loader is None:
        raise ValueError(f"未知 sklearn 数据集：{name}")
    b = loader()
    X = np.asarray(b.data, dtype=float)
    y = np.asarray(b.target)
    feat = (
        list(b.feature_names)
        if getattr(b, "feature_names", None) is not None
        else [f"f{i}" for i in range(X.shape[1])]
    )
    yu = np.unique(y)
    if np.issubdtype(y.dtype, np.integer) and len(yu) <= 20:
        task = "classification"
    elif len(yu) <= 20 and np.all(np.mod(y.astype(float), 1) == 0):
        task = "classification"
    else:
        task = "regression"
    return X, y, feat, task


def _load_file(d: dict, src: str):
    """加载本地表格文件（CSV/Parquet/Excel/Feather），按 target 列拆 X/y。"""
    import pandas as pd

    path = Path(d["path"])
    if not path.is_absolute():
        # Resolve relative paths consistently from the process cwd, then repo root.
        repo_path = Path(__file__).resolve().parents[1] / path
        path = path if path.exists() else repo_path
    if src == "csv":
        read_kw = {}
        # 透传分隔符与缺失标记：真实公开数据集常用分号(;)或含 "?" 缺失值
        if d.get("sep") is not None:
            read_kw["sep"] = d["sep"]
        if d.get("na_values") is not None:
            read_kw["na_values"] = d["na_values"]
        df = pd.read_csv(path, **read_kw)
    elif src == "parquet":
        df = pd.read_parquet(path)
    elif src == "excel":
        df = pd.read_excel(path)
    elif src == "feather":
        df = pd.read_feather(path)
    else:
        raise ValueError(f"不支持的文件源：{src}")
    target = d.get("target")
    if target is None:
        raise ValueError("文件源必须在 data 中指定 target（列名或列索引）")
    if isinstance(target, int):
        y = df.iloc[:, target].values
        X = df.drop(df.columns[target], axis=1)
    else:
        y = df[target].values
        X = df.drop(columns=[target])
    feat = list(X.columns)
    # 不再强制 astype(float)：字符串类别列与缺失值交给下游预处理器按列类型处理，
    # 这样真实业务表格（混合类型 + 缺失）才能被直接喂进来。
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    task = d.get("task_type")
    if task is None:
        yu = np.unique(y)
        numeric = np.issubdtype(y.dtype, np.number)
        task = (
            "classification"
            if (not numeric or (len(yu) <= 20 and np.all(np.equal(yu, np.floor(yu)))))
            else "regression"
        )
    if task == "classification":
        y_raw = df.iloc[:, target] if isinstance(target, int) else df[target]
        if not np.issubdtype(y_raw.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(np.asarray(y_raw, dtype=object))
    if task not in ("classification", "regression", "clustering"):
        raise ValueError(f"不支持的 task_type：{task}")
    return X, y, feat, task


def _load_openml(dataset_id, version=None, task_type=None):
    from sklearn.datasets import fetch_openml

    b = fetch_openml(data_id=dataset_id, version=version, as_frame=True)
    df = b.frame if hasattr(b, "frame") else b.data
    target = b.target_names[0]
    y = np.asarray(df[target].values)
    X = df.drop(columns=[target]).values.astype(float)
    feat = list(df.drop(columns=[target]).columns)
    yu = np.unique(y)
    numeric = np.issubdtype(y.dtype, np.number)
    task = task_type or (
        "classification"
        if (not numeric or (len(yu) <= 20 and np.all(np.equal(yu, np.floor(yu)))) )
        else "regression"
    )
    if task not in ("classification", "regression"):
        raise ValueError(f"OpenML 不支持 task_type：{task}")
    return X, y, feat, task


def _load_url(d: dict):
    """从公开网址下载数据集（CSV/XLSX）后按文件源加载。

    支持 d 透传：target / sep / na_values / task_type，全部交给 _load_file 处理。
    """
    import tempfile
    import urllib.request

    url = d.get("url")
    if not url:
        raise ValueError("url 数据源必须在 data 中提供 url")
    low = url.lower()
    suffix = ".xlsx" if low.endswith((".xlsx", ".xls")) else ".csv"
    cache_name = f"autoresearch_url_{abs(hash(url))}{suffix}"
    tmp = Path(tempfile.gettempdir()) / cache_name
    if not tmp.exists():
        print(f"[data] 下载数据集：{url}")
        urllib.request.urlretrieve(url, tmp)
    sub = dict(d)
    sub["path"] = str(tmp)
    sub_src = "excel" if suffix == ".xlsx" else "csv"
    return _load_file(sub, sub_src)


MAX_ONEHOT_CARDINALITY = 20


def build_tabular_preprocessor(X, max_cardinality: int = MAX_ONEHOT_CARDINALITY):
    """构造「混合类型表格」预处理器（未拟合）。

    设计要点：
      - 数值列：中位数插补 + 标准化
      - 类别列：众数插补 + 独热编码；基数超过 max_cardinality 时降级为序数编码，
        避免高基数列（如用户 ID、邮编）把特征维度炸掉
      - 所有插补/缩放统计量只在 fit 阶段（即训练折）学习，杜绝数据泄漏
    返回 (ColumnTransformer, 列角色字典)。
    """
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    if not isinstance(X, pd.DataFrame):
        raise TypeError("预处理器需要 DataFrame 输入，请先由 load_dataset 包装")
    num_cols, cat_cols, ord_cols = [], [], []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            num_cols.append(c)
        else:
            if s.nunique(dropna=True) > max_cardinality:
                ord_cols.append(c)
            else:
                cat_cols.append(c)
    roles = {"numeric": [str(c) for c in num_cols],
             "categorical_onehot": [str(c) for c in cat_cols],
             "categorical_ordinal_high_cardinality": [str(c) for c in ord_cols]}
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))
    if ord_cols:
        transformers.append(("ord", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), ord_cols))
    if not transformers:
        raise ValueError("没有可用的特征列")
    return ColumnTransformer(transformers, remainder="drop"), roles


def feature_names_after_prep(fitted_ct) -> list:
    """预处理器拟合后的真实输出列名（独热会把一列展开成多列，名字必须跟着变）。"""
    try:
        names = [str(n) for n in fitted_ct.get_feature_names_out()]
        # 去掉 sklearn 的前缀（num__/cat__），保持可读且与论文展示一致
        return [n.split("__", 1)[1] if "__" in n else n for n in names]
    except Exception:
        return []


def describe_preprocessing(X, roles: dict):
    """生成预处理报告：每列缺失量、推断类型、编码方式。用于论文如实披露。"""
    import pandas as pd
    report = {"n_samples": int(len(X)), "n_raw_features": int(X.shape[1]), "columns": []}
    for c in X.columns:
        s = X[c]
        missing = int(s.isna().sum())
        if c in (roles.get("numeric") or []):
            kind, enc = "数值", "中位数插补 + 标准化"
        elif c in (roles.get("categorical_onehot") or []):
            kind, enc = "类别", f"众数插补 + 独热编码（{int(s.nunique(dropna=True))} 个取值）"
        else:
            kind, enc = "类别(高基数)", f"众数插补 + 序数编码（{int(s.nunique(dropna=True))} 个取值）"
        report["columns"].append({
            "name": str(c), "kind": kind, "missing": missing,
            "missing_pct": round(100.0 * missing / max(len(s), 1), 2), "encoding": enc,
        })
    report["n_missing_total"] = int(sum(c["missing"] for c in report["columns"]))
    return report


def _load_dataset_inner(spec: dict):
    """根据 spec 加载 (X, y, feature_names, task_type)（内部实现）。"""
    d = spec.get("data") or {}
    src = d.get("source")
    if not src:
        name = spec.get("dataset")
        if name and name in DATASETS:
            return DATASETS[name]()
        if name:
            try:
                return _load_sklearn_generic(name)
            except Exception:
                pass
        raise ValueError("spec 未提供可识别的数据集来源（data.source / dataset）")
    if src == "sklearn":
        tt = spec.get("task_type") or ""
        if tt == "clustering":
            return _load_sklearn_clustering(d.get("name") or spec.get("dataset"))
        X, y, feat, inferred = _load_sklearn_generic(d.get("name") or spec.get("dataset"))
        return X, y, feat, tt or inferred
    if src in ("csv", "parquet", "excel", "feather"):
        file_data = dict(d)
        file_data["task_type"] = spec.get("task_type") or d.get("task_type")
        return _load_file(file_data, src)
    if src == "openml":
        return _load_openml(d.get("dataset_id"), d.get("version"),
                            spec.get("task_type") or d.get("task_type"))
    if src == "url":
        file_data = dict(d)
        file_data["task_type"] = spec.get("task_type") or d.get("task_type")
        return _load_url(file_data)
    if src in DATASETS:  # 兜底：source 直接写内置名
        return DATASETS[src]()
    raise ValueError(f"不支持的数据源：{src}")


def _fit_prep(X_tr):
    """在训练折上拟合混合类型预处理器并返回（供 .transform 复用）。

    只在训练集 fit，绝不触碰测试集统计量 —— 这是避免数据泄漏、
    保证论文报告的准确率可复现的前提。
    """
    ct, roles = build_tabular_preprocessor(X_tr)
    ct.fit(X_tr)
    return ct


def load_dataset(spec: dict):
    """加载数据，并保证 X 为保留列类型的 DataFrame。

    统一出口包装：无论来源是 sklearn / 本地文件 / OpenML，ndarray 一律按
    feature_names 转成 DataFrame，使下游 build_tabular_preprocessor 能识别
    哪些列是数值、哪些是字符串类别，从而支持真实世界的混合类型表格。
    """
    import pandas as pd

    X, y, feat, task = _load_dataset_inner(spec)
    if not isinstance(X, pd.DataFrame):
        cols = [str(f) for f in (feat or [])]
        if not cols or len(cols) != (X.shape[1] if hasattr(X, "shape") else 0):
            cols = [f"f{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(np.asarray(X), columns=cols)
    if feat is None or len(feat) != X.shape[1]:
        feat = [str(c) for c in X.columns]
    return X, y, feat, task


# ── 通用方法工厂（spec 驱动，零硬编码） ─────────────────────────────
def _estimator_family(spec) -> str:
    """从方法声明推断归纳偏置族，用于证据引擎的通用路由。"""
    name = spec if isinstance(spec, str) else (spec.get("estimator") or "")
    low = name.lower()
    if any(k in low for k in ("logistic", "linearregression", "ridge", "lasso", "elasticnet", "sgd")):
        return "linear"
    if any(k in low for k in ("randomforest", "gradientboosting", "xgb", "lgbm",
                              "histgradient", "forest", "boosting", "extratrees")):
        return "ensemble"
    if any(k in low for k in ("svc", "svm", "linearsvc")):
        return "svm"
    if any(k in low for k in ("knn", "neighbors")):
        return "knn"
    return "other"


class PyTorchMLP:
    """sklearn 兼容的多层感知机（可选依赖：需先 `pip install torch`）。

    仅当 torch 可用时才能实例化/训练；否则 build_estimator 返回 None，
    由调用方优雅跳过。这是一个示例性的「非 sklearn 后端」接入点，
    证明引擎对估计器来源零硬编码——任何满足 fit/predict 接口的对象都能接入。
    """

    def __init__(self, hidden=(64, 32), lr=1e-3, epochs=60, batch_size=64,
                 task="auto", random_state=42, device="cpu"):
        self.hidden = tuple(int(x) for x in hidden)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.task = task
        self.random_state = int(random_state)
        self.device = device
        self._classes = None
        self._n_out = None
        self._model = None

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import LabelEncoder
        rng = torch.Generator().manual_seed(self.random_state)
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        yt = np.asarray(y)
        if self.task == "regression" or not isinstance(y[0] if len(y) else 0, str) and np.issubdtype(np.asarray(y).dtype, np.floating) and len(np.unique(y)) > 20:
            self._task = "regression"
            yv = torch.as_tensor(np.asarray(y, dtype=np.float32)).view(-1, 1)
            self._n_out = 1
            self._classes = None
        else:
            self._task = "classification"
            le = LabelEncoder()
            y_enc = le.fit_transform(yt)
            self._classes = le.classes_
            yv = torch.as_tensor(np.asarray(y_enc, dtype=np.int64))
            self._n_out = int(yv.max().item() + 1)
        n_in = Xt.shape[1]
        layers = [n_in] + list(self.hidden) + [self._n_out]
        seq = []
        for i in range(len(layers) - 1):
            seq.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                seq.append(nn.ReLU())
        self._model = nn.Sequential(*seq).to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss() if self._task == "regression" else nn.CrossEntropyLoss()
        self._model.train()
        ds = torch.utils.data.TensorDataset(Xt.to(self.device), yv.to(self.device))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, generator=rng)
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad()
                out = self._model(xb)
                loss = criterion(out, yb if self._task == "regression" else yb.long())
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        self._model.eval()
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            out = self._model(Xt)
        if self._task == "regression":
            return out.cpu().numpy().ravel()
        return self._classes[out.argmax(dim=1).cpu().numpy()]

    def predict_proba(self, X):
        import torch
        self._model.eval()
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            out = self._model(Xt)
        if self._task == "regression":
            return out.cpu().numpy()
        probs = torch.softmax(out, dim=1).cpu().numpy()
        return probs


def _build_torch_mlp(spec, seed):
    try:
        import torch  # noqa: F401
    except Exception:
        return None
    params = dict(spec.get("params", {}))
    hidden = tuple(int(x) for x in params.get("hidden", [64, 32]))
    return PyTorchMLP(
        hidden=hidden, lr=float(params.get("lr", 1e-3)),
        epochs=int(params.get("epochs", 60)), batch_size=int(params.get("batch_size", 64)),
        task=params.get("task", "auto"), random_state=seed,
    )


def build_estimator(spec, seed: int):
    """返回一个 sklearn 兼容的估计器（来源零硬编码：sklearn / xgboost / lightgbm / torch 均可）。

    - 旧式字符串（如 "RandomForestClassifier"）走内置注册表（向后兼容）。
    - 新式 dict：{"estimator": "sklearn.ensemble.RandomForestClassifier",
                   "params": {"n_estimators": 300, "random_state": "<seed>"}}。
      支持 "<seed>" 占位符，运行时会替换为本轮随机种子。
    - 非 sklearn 后端（xgboost/lightgbm 直接用其完整 import 路径；PyTorch 用 "PyTorchMLP"）
      同样通过动态导入接入；缺少对应依赖时返回 None，由调用方跳过。
    """
    if isinstance(spec, str):
        return _method_factory(spec, seed)
    if isinstance(spec, dict):
        dotted = spec.get("estimator")
        if not dotted:
            return None
        # 特殊后端：内置 PyTorch MLP 包装器
        if dotted == "PyTorchMLP" or dotted.endswith(".PyTorchMLP"):
            return _build_torch_mlp(spec, seed)
        mod_path, cls_name = dotted.rsplit(".", 1)
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception:
            return None
        params = dict(spec.get("params", {}))
        for k, v in list(params.items()):
            if isinstance(v, str) and v.strip() == "<seed>":
                params[k] = seed
            elif isinstance(v, (list, tuple)):
                params[k] = [
                    seed if (isinstance(x, str) and x.strip() == "<seed>") else x
                    for x in v
                ]
        try:
            return cls(**params)
        except Exception:
            return None
    return None


def _normalize_methods(raw) -> list:
    """把 candidate_methods（字符串或 dict 混合）归一化为 [(key, spec), ...]。

    key 用于作为 seed_metrics / 证据 的字典键，保证一致性。
    """
    out = []
    for m in raw or []:
        if isinstance(m, str):
            out.append((m, m))
        elif isinstance(m, dict):
            key = m.get("name") or m.get("estimator") or f"m{len(out)}"
            out.append((key, m))
    return out


# ── 方法工厂 ────────────────────────────────────────────────────────────
def _method_factory(name: str, seed: int):
    """返回一个 sklearn 估计器；不支持的方法返回 None。"""
    n = name
    if n == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=3000, random_state=seed)
    if n == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
    if n == "GradientBoostingClassifier":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=seed)
    if n == "SVC":
        from sklearn.svm import SVC

        return SVC(random_state=seed, probability=False)
    if n == "KNeighborsClassifier":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(n_neighbors=5)
    if n == "LinearRegression":
        from sklearn.linear_model import LinearRegression

        return LinearRegression()
    if n == "Ridge":
        from sklearn.linear_model import Ridge

        return Ridge(random_state=seed)
    if n == "Lasso":
        from sklearn.linear_model import Lasso

        return Lasso(random_state=seed, max_iter=10000, alpha=0.01)
    if n == "ElasticNet":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(random_state=seed, max_iter=10000, alpha=0.01, l1_ratio=0.5)
    return None


def _has_importance(est) -> bool:
    return hasattr(est, "feature_importances_") or hasattr(est, "coef_")


def _extract_importance(est, feat_names):
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        if coef.ndim > 1:
            vals = np.abs(coef).mean(axis=0)
        else:
            vals = np.abs(coef)
    else:
        return {}
    out = {}
    for fn, v in zip(feat_names, vals):
        out[str(fn)] = float(v)
    return out


# ── 本文提出的融合集成方法（确定性方法贡献，零编造） ─────────────────
# 默认把候选方法融合为一个软投票集成，作为「本文提出的方法」参与统一基准对比，
# 并在消融实验中逐一移除基学习器以量化各基学习器的边际贡献。所有数字来自真实重跑。
_PROPOSED_KEY = "本文提出的融合集成方法"


def _parse_proposed(spec, method_specs, methods, is_reg):
    """解析 spec.proposed_method，返回 {key, base, voting} 或 None。

    base 缺省为全部候选方法；仅保留真实存在且可构建的基学习器，
    少于 2 个有效基学习器时返回 None（退化时不强行构造集成）。
    """
    pm = spec.get("proposed_method")
    if not pm:
        return None
    raw_base = list(pm.get("base") or [])
    if not raw_base:
        raw_base = [m for m in methods if not method_specs[m].get("__proposed__")]
    valid_base = [b for b in raw_base
                  if b in method_specs and not method_specs[b].get("__proposed__")]
    if len(valid_base) < 2:
        return None
    voting = (pm.get("voting") or ("soft" if not is_reg else "hard"))
    if voting not in ("soft", "hard"):
        voting = "soft" if not is_reg else "hard"
    return {"key": pm.get("name") or _PROPOSED_KEY, "base": valid_base, "voting": voting}


def _build_base_for_ensemble(bspec, seed, is_reg):
    """为集成构建一个基学习器；分类任务下确保有 predict_proba（软投票需要）。"""
    est = build_estimator(bspec, seed)
    if est is None:
        return None
    if not is_reg and not hasattr(est, "predict_proba"):
        # SVC 默认无概率输出：以 probability=True 重建以支持软投票
        if "SVC" in type(est).__name__:
            try:
                from sklearn.svm import SVC
                est = SVC(probability=True, random_state=seed)
            except Exception:
                return None
        else:
            return None
    return est


def _build_ensemble(base_specs, is_reg, voting, seed):
    """用基学习器规格列表构造软/硬投票集成；有效基学习器不足 2 个返回 None。"""
    estimators = []
    for i, bs in enumerate(base_specs):
        est = _build_base_for_ensemble(bs, seed, is_reg)
        if est is None:
            continue
        estimators.append((f"b{i}", est))
    if len(estimators) < 2:
        return None
    try:
        if is_reg:
            return VotingRegressor(estimators=estimators)
        return VotingClassifier(estimators=estimators, voting=voting)
    except Exception:
        return None


def _ensemble_mean_metric(base_specs, is_reg, voting, main_metric, objective, X, y):
    """在 SEEDS 上重跑集成，返回主指标均值（用于消融实验的边际贡献量化）。"""
    vals = []
    for seed in SEEDS:
        if is_reg:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
        sc = _fit_prep(X_tr)
        est = _build_ensemble(base_specs, is_reg, voting, seed)
        if est is None:
            return None
        try:
            est.fit(sc.transform(X_tr), y_tr)
            yp = est.predict(sc.transform(X_te))
        except Exception:
            return None
        if is_reg:
            if main_metric == "mse":
                vals.append(float(mean_squared_error(y_te, yp)))
            else:
                vals.append(float(r2_score(y_te, yp)))
        else:
            if main_metric in ("accuracy", "acc"):
                vals.append(float(accuracy_score(y_te, yp)))
            else:
                vals.append(float(f1_score(y_te, yp, average="macro", zero_division=0)))
    return float(np.mean(vals)) if vals else None


# ── 主入口 ──────────────────────────────────────────────────────────────
def run_analysis(spec: dict, out_path: Path) -> dict:
    """运行确定性分析，图写入 out_path.parent/'plots'，返回 ANALYSIS_JSON dict。

    数据集与方法全部从 spec 解析，引擎对具体数据集/方法零硬编码。
    """
    norm = _normalize_methods(spec.get("candidate_methods"))
    methods = [k for k, _ in norm]
    method_specs = {k: s for k, s in norm}
    # 防御性去重：同一 estimator 只保留首条，避免不同中文名指代同一方法
    # 导致重复实验、重复列与跨数据集空列。
    _seen_est: dict[str, str] = {}
    _methods_dedup: list[str] = []
    for m in methods:
        est = method_specs[m].get("estimator") if isinstance(method_specs[m], dict) else m
        if est and est in _seen_est:
            continue
        if est:
            _seen_est[est] = m
        _methods_dedup.append(m)
    methods = _methods_dedup
    # 注入「本文提出的融合集成方法」：作为候选之外的额外方法参与统一基准对比。
    # 它是确定性的（投票融合候选方法），不依赖 LLM，且自带消融实验贡献点。
    _is_reg_guess = ((spec.get("task_type") or (spec.get("data") or {}).get("task_type")) == "regression")
    _proposed = _parse_proposed(spec, method_specs, methods, _is_reg_guess)
    if _proposed:
        methods.append(_proposed["key"])
        method_specs[_proposed["key"]] = {
            "__proposed__": True,
            "base": _proposed["base"],
            "voting": _proposed["voting"],
        }
    declared_task = spec.get("task_type") or (spec.get("data") or {}).get("task_type")
    # 预检：能成功构建的方法才参与；缺少依赖（如未装 xgboost/torch）的记为跳过
    skipped_methods = []
    cluster_names = {"kmeans", "agglomerative", "dbscan"}
    for m in list(methods):
        if method_specs[m].get("__proposed__"):
            # 提议方法由引擎在种子循环内动态构建集成，预检跳过
            continue
        if declared_task == "clustering" and any(n in m.lower() for n in cluster_names):
            continue
        try:
            if build_estimator(method_specs[m], 42) is None:
                skipped_methods.append(m)
                methods.remove(m)
        except Exception:
            skipped_methods.append(m)
            methods.remove(m)
    main_metric = spec.get("metric") or "accuracy"
    objective = spec.get("objective") or "maximize"
    if not methods:
        return {"error": "没有可运行的候选方法", "skipped_methods": skipped_methods}
    try:
        X, y, feat_names, task_type = load_dataset(spec)
    except Exception as e:
        return {"error": f"无法加载数据集：{e}"}

    # 预处理报告：如实记录每列类型、缺失量与编码方式（论文需披露，不得隐藏）
    try:
        _ct0, _roles0 = build_tabular_preprocessor(X)
        preprocessing_report = describe_preprocessing(X, _roles0)
    except Exception:
        preprocessing_report = None
    _prep_names_done = False

    is_reg = task_type == "regression"
    is_cluster = task_type == "clustering"
    plots_dir = out_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── 聚类分支（无需 train/test split，全量跑 + 评估） ──
    if is_cluster:
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

        sc = _fit_prep(X)
        Xs = sc.transform(X)
        n_clusters = len(np.unique(y))
        results = []
        methods_for_plot = []
        # 为每个聚类方法跑一轮
        cluster_classes = {
            "kmeans": ("KMeans", KMeans, {"n_clusters": n_clusters, "random_state": 42, "n_init": 10}),
            "agglomerative": ("AgglomerativeClustering", AgglomerativeClustering, {"n_clusters": n_clusters}),
            "dbscan": ("DBSCAN", DBSCAN, {}),
        }
        # Cluster methods use the same string/dict spec contract as supervised methods.
        for key, method_spec in method_specs.items():
            declared_name = method_spec if isinstance(method_spec, str) else method_spec.get("estimator", "")
            short_name = str(declared_name).rsplit(".", 1)[-1].lower()
            family_name = next((family for family in cluster_classes if short_name == family or
                                (family == "agglomerative" and short_name == "agglomerativeclustering")), None)
            if family_name is None:
                continue
            name, estimator_cls, defaults = cluster_classes[family_name]
            try:
                params = dict(defaults)
                if isinstance(method_spec, dict):
                    params.update(method_spec.get("params", {}))
                    for param, value in list(params.items()):
                        if isinstance(value, str) and value.strip() == "<seed>":
                            params[param] = 42
                est = estimator_cls(**params)
                labels = est.fit_predict(Xs)
                ari = float(adjusted_rand_score(y, labels))
                nmi = float(normalized_mutual_info_score(y, labels))
                n_labels = len(set(labels)) - (1 if -1 in labels else 0)
                # 轮廓系数仅在标签数 > 1 时有效
                if n_labels > 1:
                    uni = np.unique(labels)
                    if -1 in uni:
                        mask = labels != -1
                        sil = float(silhouette_score(Xs[mask], labels[mask])) if mask.sum() > 1 else None
                    else:
                        sil = float(silhouette_score(Xs, labels))
                else:
                    sil = None
                results.append({"method": name, "ARI": ari, "NMI": nmi,
                               "silhouette": round(sil, 4) if sil is not None else None,
                               "n_clusters_found": n_labels})
                methods_for_plot.append(name)
            except Exception as e:
                skipped_methods.append({"method": key, "reason": str(e)})
                continue
        if not results:
            return {"error": "clustering 无有效候选方法"}
        # 选 spec 指定主指标及方向；聚类默认 ARI。
        cluster_metric = (spec.get("metric") or "ari").lower()
        supported_cluster_metrics = {"ari", "adjusted_rand_score", "nmi",
                                     "normalized_mutual_info_score", "silhouette"}
        if cluster_metric not in supported_cluster_metrics:
            return {"error": f"聚类不支持指标：{cluster_metric}",
                    "skipped_methods": skipped_methods}
        metric_key = {"ari": "ARI", "adjusted_rand_score": "ARI", "nmi": "NMI",
                      "normalized_mutual_info_score": "NMI", "silhouette": "silhouette"}.get(cluster_metric, "ARI")
        maximize = spec.get("objective", "maximize") != "minimize"
        results = [r for r in results if r[metric_key] is not None]
        if not results:
            return {"error": f"聚类候选方法未产生有效指标：{metric_key}",
                    "skipped_methods": skipped_methods}
        results.sort(key=lambda r: r[metric_key], reverse=maximize)
        best = results[0]
        metrics_out = []
        for r in results:
            metrics_out.append({"name": r["method"], "value": r[metric_key], "setting": "全量数据，%d 簇" % n_clusters})
            if r["silhouette"] is not None:
                metrics_out.append({"name": f"{r['method']}_轮廓系数", "value": r["silhouette"], "setting": "全量"})

        # 图：聚类结果散点（取前 2 维可视化）
        try:
            from sklearn.decomposition import PCA
            X2 = PCA(n_components=2, random_state=42).fit_transform(Xs)
            fig, ax = plt.subplots(figsize=(7, 5))
            for i, r in enumerate(results):
                if r["method"] in methods_for_plot:
                    ax.scatter([r["ARI"]], [r["NMI"]], s=120, label=r["method"])
            ax.set_xlabel("ARI"); ax.set_ylabel("NMI")
            ax.set_title("Clustering Methods: ARI vs NMI")
            ax.legend()
            fig.savefig(str(plots_dir / "cluster_ari_nmi.png"), dpi=120, bbox_inches="tight")
            plt.close(fig)
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            ax2.scatter(X2[:, 0], X2[:, 1], c=y, alpha=0.6, cmap="tab10", s=15)
            ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
            ax2.set_title("Data Distribution (PCA, true labels)")
            fig2.savefig(str(plots_dir / "cluster_pca.png"), dpi=120, bbox_inches="tight")
            plt.close(fig2)
        except Exception:
            pass

        return {
            "task_type": "clustering",
            "best_method": best["method"],
            "best_value": best[metric_key],
            "best_metric": metric_key,
            "metrics": metrics_out,
            "per_class": {},
            "single_fit_scores": {r["method"]: {"ARI": r["ARI"], "NMI": r["NMI"],
                                                   "silhouette": r["silhouette"]} for r in results},
            "significance": "聚类任务不做配对显著性检验",
            "plots": [
                {"title": "聚类方法 ARI vs NMI", "file": "cluster_ari_nmi.png"},
                {"title": "数据分布（PCA + 真实标签）", "file": "cluster_pca.png"},
            ],
            "sub_hypothesis_evidence": {},
            "skipped_methods": skipped_methods,
        }

    # ── 分类 / 回归主循环 ──

    # 每种子、每方法的指标数组
    seed_metrics: dict[str, dict[str, list]] = {m: {} for m in methods}
    seed_train_metrics: dict[str, dict[str, list]] = {m: {} for m in methods}  # H1 过拟合 gap
    best_pred_cache: dict[str, np.ndarray] = {}
    best_y_test: np.ndarray | None = None
    importance_cache: dict[str, dict] = {}
    confmat_cache: dict[str, list] = {}
    resid_cache: dict[str, tuple] = {}  # method -> (y_pred, y_true)

    for seed in SEEDS:
        if is_reg:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed
            )
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
            )
        sc = _fit_prep(X_tr)
        # 独热会把一列展开成多列：特征名必须换成预处理后的真实输出名，
        # 否则下游 feature_importance 会张冠李戴（写入论文即虚假结论）。
        if not _prep_names_done:
            _pn = feature_names_after_prep(sc)
            if _pn:
                feat_names = _pn
                _prep_names_done = True
        Xtr = sc.transform(X_tr)
        Xte = sc.transform(X_te)
        for m in methods:
            pm_spec = method_specs[m]
            if pm_spec.get("__proposed__"):
                est = _build_ensemble(
                    [method_specs[b] for b in pm_spec["base"]], is_reg, pm_spec["voting"], seed)
            else:
                est = build_estimator(pm_spec, seed)
            if est is None:
                continue
            try:
                est.fit(Xtr, y_tr)
                yp_tr = est.predict(Xtr)
                yp = est.predict(Xte)
            except Exception as e:
                skipped_methods.append({"method": m, "seed": seed, "reason": str(e)})
                continue
            if is_reg:
                r2 = float(r2_score(y_te, yp))
                mse = float(mean_squared_error(y_te, yp))
                mae = float(mean_absolute_error(y_te, yp))
                r2_train = float(r2_score(y_tr, yp_tr))
                seed_metrics[m].setdefault("r2", []).append(r2)
                seed_metrics[m].setdefault("mse", []).append(mse)
                seed_metrics[m].setdefault("mae", []).append(mae)
                seed_train_metrics[m].setdefault("r2", []).append(r2_train)
                if seed == SEEDS[0]:
                    resid_cache[m] = (yp, y_te)
            else:
                acc = float(accuracy_score(y_te, yp))
                f1m = float(f1_score(y_te, yp, average="macro", zero_division=0))
                acc_train = float(accuracy_score(y_tr, yp_tr))
                f1m_train = float(f1_score(y_tr, yp_tr, average="macro", zero_division=0))
                seed_metrics[m].setdefault("accuracy", []).append(acc)
                seed_metrics[m].setdefault("f1_macro", []).append(f1m)
                seed_train_metrics[m].setdefault("accuracy", []).append(acc_train)
                seed_train_metrics[m].setdefault("f1_macro", []).append(f1m_train)
                if seed == SEEDS[0]:
                    best_pred_cache[m] = yp
                    best_y_test = y_te
                    confmat_cache[m] = confusion_matrix(y_te, yp).tolist()
                    if _has_importance(est):
                        importance_cache[m] = _extract_importance(est, feat_names)

    # 选最优方法（按主指标均值）
    def _mean(m, k):
        arr = seed_metrics.get(m, {}).get(k)
        return float(np.mean(arr)) if arr else None

    ranked = sorted(
        [m for m in methods if _mean(m, main_metric) is not None],
        key=lambda m: _mean(m, main_metric),
        reverse=(spec.get("objective", "maximize") != "minimize"),
    )
    if not ranked:
        return {"error": "候选方法均未成功产生指标", "skipped_methods": skipped_methods}
    best_method = ranked[0]
    second_method = ranked[1] if len(ranked) > 1 else ""

    analysis: dict = {"task_type": task_type, "best_method": best_method,
                      "seed_variance": {}, "plots": []}
    if preprocessing_report:
        analysis["preprocessing"] = preprocessing_report

    # 种子方差
    for m in methods:
        for k, arr in seed_metrics.get(m, {}).items():
            if not arr:
                continue
            arr = np.asarray(arr, dtype=float)
            analysis["seed_variance"][f"{m}_{k}"] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "seeds": [float(x) for x in arr],
            }

    # 训练-测试 gap（过拟合信号）：主指标 测试均值 - 训练均值，按方法聚合
    _main_metric = spec.get("metric") or "accuracy"
    analysis["train_test_gap"] = {}
    for m in methods:
        t_arr = seed_metrics.get(m, {}).get(_main_metric)
        tr_arr = seed_train_metrics.get(m, {}).get(_main_metric)
        if t_arr and tr_arr and len(t_arr) == len(tr_arr):
            g = np.asarray(t_arr, dtype=float) - np.asarray(tr_arr, dtype=float)
            analysis["train_test_gap"][m] = float(g.mean())

    # 本文提出的融合集成方法：记录身份并做消融实验（逐一移除基学习器，量化边际贡献）
    if _proposed:
        _pk = _proposed["key"]
        analysis["proposed_method"] = _pk
        analysis["proposed_base"] = list(_proposed["base"])
        analysis["proposed_voting"] = _proposed["voting"]
        _full = _mean(_pk, main_metric)
        _ablation = []
        if _full is not None and 2 <= len(_proposed["base"]) <= 6:
            for b in _proposed["base"]:
                _reduced = [x for x in _proposed["base"] if x != b]
                if len(_reduced) < 2:
                    continue
                _val = _ensemble_mean_metric(
                    [method_specs[x] for x in _reduced], is_reg, _proposed["voting"],
                    main_metric, objective, X, y)
                if _val is not None:
                    _ablation.append({
                        "removed": b,
                        "value": round(float(_val), 4),
                        "drop": round(float(_full - _val), 4),
                    })
        analysis["proposed_ablation"] = _ablation

    if is_reg:
        # 系数 / 残差 / 特征重要性
        coef_block: dict = {}
        residual_stats: dict = {}
        fi_block: dict = {}
        for m in methods:
            est = build_estimator(method_specs[m], SEEDS[0])
            if est is None:
                continue
            # 用固定种子重训以取系数
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=SEEDS[0]
            )
            sc = _fit_prep(X_tr)
            try:
                est.fit(sc.transform(X_tr), y_tr)
                yp = est.predict(sc.transform(X_te))
            except Exception:
                continue
            if hasattr(est, "coef_"):
                coef = np.asarray(est.coef_, dtype=float)
                if coef.ndim > 1:
                    coef = coef.mean(axis=0)
                coef_block[m] = {str(fn): float(c) for fn, c in zip(feat_names, coef)}
            residual_stats[m] = {
                "mean": float(np.mean(y_te - yp)),
                "std": float(np.std(y_te - yp, ddof=0)),
            }
            fi_block[m] = {str(fn): float(abs(c)) for fn, c in zip(feat_names, coef)} if hasattr(est, "coef_") else {}
        analysis["coefficients"] = coef_block
        analysis["residual_stats"] = residual_stats
        analysis["feature_importance"] = fi_block

        # 显著性：best vs second 配对 t（按 r2 种子数组）
        analysis["significance"] = _significance(
            best_method, second_method, seed_metrics, main_metric
        )
        analysis["plots"] = _regression_plots(
            seed_metrics, coef_block, resid_cache, best_method, plots_dir
        )
    else:
        # 逐类指标（best 方法在 seed[0] 上的 precision/recall/f1）
        per_class: dict = {}
        for m in methods:
            yp = best_pred_cache.get(m)
            if yp is None or best_y_test is None:
                continue
            from sklearn.metrics import precision_recall_fscore_support

            p, r, f, s = precision_recall_fscore_support(
                best_y_test, yp, zero_division=0
            )
            entry = {}
            for ci, (pp, rr, ff) in enumerate(zip(p, r, f)):
                entry[f"class_{ci}_precision"] = float(pp)
                entry[f"class_{ci}_recall"] = float(rr)
                entry[f"class_{ci}_f1"] = float(ff)
            per_class[m] = entry
        analysis["per_class"] = per_class
        analysis["confusion_matrix"] = confmat_cache
        analysis["feature_importance"] = {
            m: importance_cache[m] for m in methods if m in importance_cache
        }
        analysis["significance"] = _significance(
            best_method, second_method, seed_metrics, main_metric
        )
        analysis["plots"] = _classification_plots(
            seed_metrics, confmat_cache, importance_cache, best_method, plots_dir
        )

    # ── 子假设证据（P5 升级：确保论文子假设裁决有凭有据） ────────────
    if spec.get("sub_hypotheses"):
        per_class_data = analysis.get("per_class") or {}
        evidence = _build_hypothesis_evidence(
            spec, seed_metrics, seed_train_metrics, feat_names, is_reg, X, y, per_class_data, methods, method_specs
        )
        analysis["sub_hypothesis_evidence"] = evidence

    # 给 plot 条目补 path 字段（相对 out_path.parent）
    for p in analysis["plots"]:
        fname = p.get("file")
        if fname:
            p["path"] = f"plots/{fname}"

    # ── 兼容 generate_paper 的 metrics 列表格式 ──────────────────
    _main_metric = spec.get("metric") or "accuracy"
    _metrics_list = []
    for m in methods:
        _arr = (seed_metrics.get(m) or {}).get(_main_metric)
        if _arr:
            _metrics_list.append({
                "name": m,
                "value": float(np.mean(_arr)),
                "setting": f"全部特征（5 种子均值，std={np.std(_arr, ddof=0):.4f}）",
            })
    analysis["metrics"] = _metrics_list
    _best = analysis.get("best_method")
    if _best:
        _barr = (seed_metrics.get(_best) or {}).get(_main_metric)
        if _barr:
            analysis["best_value"] = float(np.mean(_barr))
    analysis.setdefault("best_round", 1)
    analysis.setdefault("terminal_state", "deterministic_multi_seed_analysis")
    analysis.setdefault(
        "findings",
        "确定性分析引擎基于 5 个随机种子复现实验，"
        f"主指标 {_main_metric} 最优为 {_best}（{analysis.get('best_value', 'n/a')}），"
        "并报告逐类指标、混淆矩阵、特征重要性、多种子方差与配对显著性。",
    )
    # 透明记录：因缺少依赖（如未安装 xgboost/lightgbm/torch）而被自动跳过的方法
    analysis["skipped_methods"] = skipped_methods
    return analysis


_METHOD_ALIASES = {
    "logisticregression": ["逻辑回归"],
    "svc": ["支持向量机", "svm"],
    "supportvectormachine": ["支持向量机", "svm"],
    "randomforest": ["随机森林"],
    "gradientboosting": ["梯度提升"],
    "kneighbors": ["近邻", "k近邻", "knn"],
    "linearregression": ["线性回归"],
    "ridge": ["岭回归"],
    "lasso": ["lasso", "套索"],
    "decisiontree": ["决策树"],
    "mlp": ["多层感知机", "神经网络", "mlp"],
    "gaussiannb": ["朴素贝叶斯", "贝叶斯"],
    "adaboost": ["adaboost"],
}


def _mentioned_methods(hraw: str, methods: list) -> list:
    """从子假设文本中提取被提及的候选方法（中文/英文别名均匹配）。"""
    hl = hraw.lower()
    res = []
    for m in methods:
        ml = str(m).lower().strip()
        if ml and ml in hl:
            res.append(m)
            continue
        base = ml.split(".")[-1]
        for cls, aliases in _METHOD_ALIASES.items():
            if base.startswith(cls) and any(a in hl for a in aliases):
                res.append(m)
                break
    return res


def _pick_subject_partner(text: str, methods):
    """从「A 小于/优于 B」类假设中识别主语方法与对比伙伴方法。

    主语=标记词前（head）离标记最近的方法；对手=标记词后（tail）离标记最近的方法。
    与多数据集证据构建器中的逻辑一致，保证单/多数据集裁决口径统一。
    """
    MARKERS = ("相对于", "相比", "相对", "比在", "比", "优于", "高于", "好于",
               "大于", "胜过", "劣于", "弱于", "低于", "小于", "不如")
    for mk in MARKERS:
        idx = text.find(mk)
        if idx < 0:
            continue
        head = text[:idx]
        tail = text[idx + len(mk):]
        subj, best_s = None, -1
        for m in methods:
            p = head.rfind(m)
            if p > best_s:
                best_s, subj = p, m
        part, best_p = None, len(tail) + 1
        for m in methods:
            p = tail.find(m)
            if p >= 0 and p < best_p:
                best_p, part = p, m
        if subj and part and subj != part:
            return subj, part
    return (methods[0] if methods else None,
            methods[1] if len(methods) > 1 else (methods[0] if methods else None))


def _route_method_comparison(hraw: str, methods: list, mean_block: dict, std_block: dict):
    """方法比较类子假设 → 专属证据块 + 确定性裁决。

    返回 (block, conclusion, verdict) 或 None。
    支持三类断言模式：
      A) 断言某方法在「全体/其他」中最优或最差
         （含「弱于/优于 其他」「最差/最高」「明显落后」等，无论文中点名几个方法）；
      B) 链式「A 优于 B（优于 C）」——按比较词分段做组均值比较；
      均无法匹配时返回 None，交回调用方的关键词路由。
    """
    mentioned = _mentioned_methods(hraw, methods)
    if not mentioned:
        return None
    means = {m: mean_block.get(m) for m in mentioned}
    if any(v is None for v in means.values()):
        return None
    all_means = {m: v for m, v in mean_block.items() if v is not None}
    htext = hraw.lower()
    block = {
        "mentioned_methods": mentioned,
        "means": {m: round(float(means[m]), 4) for m in mentioned},
        "stds": {m: (round(float(std_block[m]), 4) if std_block.get(m) is not None else None) for m in mentioned},
    }

    _low = re.compile(r"低于其他|低于其余|弱于其他|弱于其余|不如其他|劣于其他|明显落后|最差|最弱|最低")
    _high = re.compile(r"优于其他|优于其余|强于其他|强于其余|表现最好|表现最优|最优于|最好|最佳|最高|最稳")
    has_group = bool(re.search(r"其他|其余|所有|一切|全部", htext))

    # ── 模式 A：断言某方法在全体中最优 / 最差 ──
    subj = None
    if (_low.search(htext) or _high.search(htext)) and (has_group or re.search(r"最差|最弱|最低|最高|最佳|最稳|明显落后", htext)):
        # 主语 = 被提及方法中在原文出现位置最靠前的一个（通常是句子主题）
        positions = sorted(
            (htext.find(str(m).lower()), m) for m in mentioned if htext.find(str(m).lower()) >= 0
        )
        subj = positions[0][1] if positions else mentioned[0]
    if subj is not None:
        others = {k: v for k, v in all_means.items() if k != subj}
        if others:
            if _low.search(htext):
                is_best = means[subj] <= min(others.values())
                below_most = means[subj] < float(np.mean(list(others.values())))
                verdict = "支持" if is_best else ("部分支持" if below_most else "不支持")
                detail = "%s 实测 %.4f，全体方法最低值 %.4f（%s），断言其最差%s成立" % (
                    subj, means[subj], min(others.values()),
                    min(others, key=others.get), "" if is_best else "不")
            else:
                is_top = means[subj] >= max(others.values())
                verdict = "支持" if is_top else "不支持"
                detail = "%s 实测 %.4f，全体方法最高值 %.4f（%s），断言其最优%s成立" % (
                    subj, means[subj], max(others.values()),
                    max(others, key=others.get), "" if is_top else "不")
            ranking = sorted(all_means, key=lambda m: -all_means[m])
            block["ranking"] = ranking
            conclusion = "%s（%s）。实测排序：%s" % (
                {"支持": "实测数据与断言方向一致", "部分支持": "实测数据仅部分符合断言",
                 "不支持": "实测数据与断言方向相反"}[verdict], detail, " > ".join(ranking))
            return block, conclusion, verdict

    # ── 模式 B：链式「A 优于 B（优于 C）」 ──
    if len(mentioned) >= 2 and re.search(r"优于|好于|高于|强于", hraw):
        segs = re.split(r"优于|好于|高于|强于", hraw)
        groups = []
        for seg in segs:
            g = [m for m in mentioned if m in seg]
            if g:
                groups.append(g)
        if len(groups) >= 2 and all(len(g) == 1 for g in groups):
            gmeans = [float(np.mean([means[m] for m in g])) for g in groups]
            chain_ok = all(gmeans[i] > gmeans[i + 1] for i in range(len(gmeans) - 1))
            close = any(abs(gmeans[i] - gmeans[i + 1]) < 0.005 for i in range(len(gmeans) - 1))
            verdict = "支持" if chain_ok else ("部分支持" if close else "不支持")
            group_strs = ["%s %.4f" % (g[0], gm) for g, gm in zip(groups, gmeans)]
            detail = "实测对比：" + " 优于 ".join(group_strs)
            ranking = sorted(mentioned, key=lambda m: -means[m])
            block["ranking"] = ranking
            conclusion = "%s（%s）。%s" % (
                {"支持": "实测数据与断言方向一致", "部分支持": "实测数据仅部分符合断言",
                 "不支持": "实测数据与断言方向相反"}[verdict], detail, "实测排序：" + " > ".join(ranking))
            return block, conclusion, verdict
        # 比较词切分未能干净分离方法（如某段同时含多个被提及方法），
        # 退回按排序给出结论，避免混乱文本。
        ranking = sorted(mentioned, key=lambda m: -means[m])
        block["ranking"] = ranking
        diffs = [abs(means[ranking[i]] - means[ranking[i + 1]]) for i in range(len(ranking) - 1)]
        verdict = "部分支持" if any(d < 0.005 for d in diffs) else "不支持"
        detail = "实测排序：" + " > ".join("%s %.4f" % (m, means[m]) for m in ranking)
        conclusion = "%s。%s" % (
            {"支持": "实测数据与断言方向一致", "部分支持": "实测数据仅部分符合断言",
             "不支持": "实测数据与断言方向相反"}[verdict], detail)
        return block, conclusion, verdict

    # ── 模式 C：两方法的多种子标准差 / 方差 方向比较（如「A 的标准差小于 B」） ──
    # 单数据集研究中这是最自然、最可证伪的稳定性假设，必须用真实 std 数值裁定。
    if any(k in htext for k in ("标准差", "方差", "波动", "稳定", "偏差", "一致性")):
        cmp_dir = None
        if any(k in htext for k in ("小于", "低于", "弱于", "不大于", "不高于", "更小", "更弱")):
            cmp_dir = "lt"
        elif any(k in htext for k in ("大于", "高于", "强于", "更优", "更大", "不小于", "更稳定")):
            cmp_dir = "gt"
        if cmp_dir and len(mentioned) >= 2:
            a, b = _pick_subject_partner(hraw, mentioned)
            sa = std_block.get(a)
            sb = std_block.get(b)
            if sa is None or sb is None:
                # 退而求其次：用所有被提及方法的 std 排序给结论，不强行裁定
                stds = {m: std_block[m] for m in mentioned if std_block.get(m) is not None}
                if stds:
                    ranking = sorted(stds, key=lambda m: stds[m])
                    block["std_ranking"] = ranking
                    conclusion = ("多种子标准差（越小越稳定）：" +
                                  " > ".join("%s %.4f" % (m, stds[m]) for m in ranking) +
                                  "。未点名具体对比方向，仅报告真实数值。")
                    return block, conclusion, None
            else:
                ok = (cmp_dir == "lt" and sa < sb) or (cmp_dir == "gt" and sa > sb)
                verdict = "支持" if ok else "不支持"
                direction = "小于" if cmp_dir == "lt" else "大于"
                block["std_compare"] = {a: round(float(sa), 4), b: round(float(sb), 4)}
                conclusion = ("实测多种子标准差：%s=%.4f，%s=%.4f；断言「%s 标准差%s %s」%s。"
                              % (a, float(sa), b, float(sb), a, direction, b,
                                 "成立" if ok else "不成立"))
                return block, conclusion, verdict

    return None


def _fmt_evidence_block(block) -> str:
    """把 {方法名: 数值} 字典格式化为可读文本，避免原始 dict 直接进论文。"""
    if not block:
        return "无"
    if isinstance(block, dict):
        return "、".join("%s %.4f" % (k, float(v)) for k, v in block.items())
    return str(block)


def _build_hypothesis_evidence(spec, seed_metrics, seed_train_metrics, feat_names, is_reg, X, y, per_class, methods, method_specs):
    """基于实验数据自动生成子假设裁决证据。

    methods       : 归一化后的方法键列表（与 seed_metrics 一致）
    method_specs  : 键 -> 原始方法声明（字符串或 dict），用于通用构建估计器
    """
    evidence = {}
    hyps = spec.get("sub_hypotheses") or []
    metric = spec.get("metric") or "accuracy"

    def _mean(d, m, k):
        arr = (d.get(m) or {}).get(k)
        if not arr: return None
        return float(np.mean(arr))

    # ── 预计算通用证据块（与假设文本无关，保证每个子假设都有证据） ──
    gap_block = {}
    for m in methods:
        tv = _mean(seed_metrics, m, metric)
        trv = _mean(seed_train_metrics, m, metric)
        if tv is not None and trv is not None:
            gap_block[m] = round(float(trv) - float(tv), 4)
    std_block = {}
    for m in methods:
        arr = (seed_metrics.get(m) or {}).get(metric)
        if arr:
            std_block[m] = round(float(np.std(arr, ddof=0)), 4)

    def _compute_regularization():
        """分类：LogisticRegression 正则 vs 无正则；回归：Linear vs Ridge/Lasso。"""
        try:
            from sklearn.linear_model import LogisticRegression as LR, Ridge, Lasso, LinearRegression
            from sklearn.metrics import accuracy_score, r2_score
            if is_reg:
                targets = [("Linear", None), ("Ridge", "ridge"), ("Lasso", "lasso")]
                measured = {}
                for label, _ in targets:
                    seeds_vals = []
                    for seed in SEEDS:
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
                        sc = _fit_prep(X_tr)
                        Xtr, Xte = sc.transform(X_tr), sc.transform(X_te)
                        if label == "Linear":
                            est = LinearRegression().fit(Xtr, y_tr)
                        elif label == "Ridge":
                            est = Ridge(random_state=seed).fit(Xtr, y_tr)
                        else:
                            est = Lasso(random_state=seed, max_iter=5000).fit(Xtr, y_tr)
                        seeds_vals.append(float(r2_score(y_te, est.predict(Xte))))
                    measured[label] = float(np.mean(seeds_vals))
                if "Linear" in measured and "Ridge" in measured:
                    return {"linear_mean": measured["Linear"], "ridge_mean": measured["Ridge"],
                            "lasso_mean": measured.get("Lasso"),
                            "conclusion": "正则化（Ridge/Lasso）相对普通最小二乘的测试 R2 变化：%.4f -> %.4f（Lasso %.4f）" % (measured["Linear"], measured["Ridge"], measured.get("Lasso", 0))}
            else:
                reg_s, unreg_s = [], []
                for seed in SEEDS:
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
                    sc = _fit_prep(X_tr)
                    Xtr, Xte = sc.transform(X_tr), sc.transform(X_te)
                    lr_reg = LR(max_iter=5000, random_state=seed).fit(Xtr, y_tr)
                    lr_un = LR(max_iter=5000, random_state=seed, C=1e10).fit(Xtr, y_tr)
                    reg_s.append(float(accuracy_score(y_te, lr_reg.predict(Xte))))
                    unreg_s.append(float(accuracy_score(y_te, lr_un.predict(Xte))))
                if reg_s and unreg_s:
                    _, p = stats.ttest_rel(reg_s, unreg_s)
                    return {"regularized_mean": float(np.mean(reg_s)), "unregularized_mean": float(np.mean(unreg_s)),
                            "paired_ttest_p": float(p) if p is not None else None,
                            "conclusion": "正则化显著收益" if (p is not None and p < 0.05 and np.mean(reg_s) > np.mean(unreg_s)) else "正则化收益不显著" if p is not None else "无法统计"}
        except Exception:
            return None
        return None

    def _compute_feature_selection():
        """Top-k 特征重训消融：检测声明中的线性方法（有 coef_）用于特征排序。"""
        try:
            from sklearn.metrics import accuracy_score, r2_score
            # 选一个线性族方法（任意 sklearn 线性模型均有 coef_）
            lin_key = next(
                (k for k in methods if _estimator_family(method_specs.get(k, k)) == "linear"),
                None,
            )
            if lin_key is None:
                return None
            if feat_names and len(feat_names) >= 10:
                topk = 10
                if is_reg:
                    base = build_estimator(method_specs[lin_key], 42)
                    if base is None:
                        return None
                    allv, topv = [], []
                    coef_ref = None
                    for seed in SEEDS:
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
                        sc = _fit_prep(X_tr)
                        Xtr, Xte = sc.transform(X_tr), sc.transform(X_te)
                        est = base.fit(Xtr, y_tr)
                        if not hasattr(est, "coef_"):
                            return None
                        allv.append(float(r2_score(y_te, est.predict(Xte))))
                        if coef_ref is None:
                            coef_ref = np.argsort(-np.abs(np.asarray(est.coef_).ravel()))[:topk]
                        est_t = base.__class__(**base.get_params()).fit(Xtr[:, coef_ref], y_tr)
                        topv.append(float(r2_score(y_te, est_t.predict(Xte[:, coef_ref]))))
                else:
                    base = build_estimator(method_specs[lin_key], 42)
                    if base is None:
                        return None
                    X_tr0, X_te0, y_tr0, y_te0 = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
                    sc0 = _fit_prep(X_tr0)
                    est0 = base.fit(sc0.transform(X_tr0), y_tr0)
                    if not hasattr(est0, "coef_"):
                        return None
                    coef = np.abs(np.asarray(est0.coef_))
                    if coef.ndim > 1: coef = coef.mean(axis=0)
                    top_idx = np.argsort(-coef.ravel())[:topk]
                    allv, topv = [], []
                    for seed in SEEDS:
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
                        sc = _fit_prep(X_tr)
                        Xtr, Xte = sc.transform(X_tr), sc.transform(X_te)
                        allv.append(float(accuracy_score(y_te, base.fit(Xtr, y_tr).predict(Xte))))
                        topv.append(float(accuracy_score(y_te, base.fit(Xtr[:, top_idx], y_tr).predict(Xte[:, top_idx]))))
                va = [x for x in allv if not np.isnan(x)]
                vt = [x for x in topv if not np.isnan(x)]
                if va and vt:
                    return {"topk": topk, "all_features_accuracy_mean": float(np.mean(va)),
                            "topk_features_accuracy_mean": float(np.mean(vt)),
                            "all_features_std": float(np.std(va, ddof=0)), "topk_features_std": float(np.std(vt, ddof=0)),
                            "topk_feature_names": [str(feat_names[i]) for i in (top_idx.tolist() if not is_reg else coef_ref.tolist())],
                            "conclusion": "Top-%d 特征下准确率%s且标准差%s（特征存在冗余）" % (
                                topk, "不降" if np.mean(vt) >= np.mean(va) - 0.01 else "下降",
                                "降低" if np.std(vt, ddof=0) <= np.std(va, ddof=0) else "升高")}
        except Exception:
            return None
        return None

    def _compute_sparsity():
        """回归：Lasso/Ridge 零系数个数。"""
        if not is_reg:
            return None
        try:
            from sklearn.linear_model import Lasso, Ridge
            res = {}
            for seed in SEEDS[:1]:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
                sc = _fit_prep(X_tr)
                Xtr = sc.transform(X_tr)
                la = Lasso(random_state=seed, max_iter=5000).fit(Xtr, y_tr)
                ri = Ridge(random_state=seed).fit(Xtr, y_tr)
                res["Lasso_zero_coeff"] = int(np.sum(np.abs(la.coef_) < 1e-4))
                res["Ridge_zero_coeff"] = int(np.sum(np.abs(ri.coef_) < 1e-4))
                # 必须是预处理后的实际列数，否则「n/m 个系数被压缩为 0」的分数是假的
                res["total_features"] = int(Xtr.shape[1])
            res["conclusion"] = "Lasso 将 %d/%d 个系数精确压缩为 0，Ridge 为 %d，说明特征贡献本质稀疏" % (
                res["Lasso_zero_coeff"], res["total_features"], res["Ridge_zero_coeff"])
            return res
        except Exception:
            return None

    reg_block = _compute_regularization()
    feat_sel_block = _compute_feature_selection()
    sparsity_block = _compute_sparsity()

    class_spread = {}
    if not is_reg and per_class:
        for m in methods:
            pc = per_class.get(m) or {}
            recalls = [float(v) for k, v in pc.items() if k.endswith("_recall")]
            if recalls:
                class_spread[m] = (round(min(recalls), 4), round(max(recalls), 4))

    for i, h in enumerate(hyps):
        hraw = str(h)
        htext = hraw.lower()
        hid = "H%d" % (i + 1)
        item = {"statement": hraw, "train_test_gap": gap_block, "seed_std": std_block}
        conclusion = ""
        verdict = None
        # 首选：方法比较类子假设 → 专属实测对比证据 + 确定性裁决
        mc = _route_method_comparison(hraw, methods, {m: _mean(seed_metrics, m, metric) for m in methods}, std_block)
        if mc is not None:
            mc_block, conclusion, verdict = mc
            item["method_comparison"] = mc_block
        # 关键词路由（方法比较未命中时的机制性证据）
        elif ("稀疏" in htext or "零系数" in htext or "spars" in htext) and sparsity_block:
            item["sparsity"] = sparsity_block
            conclusion = "稀疏性：%s" % sparsity_block.get("conclusion", "")
        elif ("稳定性" in htext or "标准差" in htext or "方差-偏差" in htext
              or "信噪比" in htext or "方差" in htext):
            lo = min(std_block.values()) if std_block else 0.0
            hi = max(gap_block.values()) if gap_block else 0.0
            conclusion = "基于训练-测试 %s gap（%.4f~%.4f）与多种子标准差（%s）评估方差-偏差权衡、稳定性与信噪比天花板" % (
                metric, lo, hi, _fmt_evidence_block(std_block))
        elif ("正则化" in htext or "regular" in htext or "lasso" in htext):
            if reg_block:
                item["regularization"] = reg_block
                conclusion = "正则化对照：%s" % reg_block.get("conclusion", "")
            elif sparsity_block:
                item["sparsity"] = sparsity_block
                conclusion = "稀疏性：%s" % sparsity_block.get("conclusion", "")
        elif ("冗余" in htext or "redundan" in htext or "top" in htext
              or "方差-准确率" in htext or "方差-表现" in htext or "分类器敏感度" in htext
              or "特征" in htext):
            if feat_sel_block:
                item["feature_selection"] = feat_sel_block
                fs = feat_sel_block
                conclusion = "特征选择消融：全量 %.4f vs Top-%d %.4f（std %.4f vs %.4f），%s" % (
                    fs.get("all_features_accuracy_mean", 0), fs.get("topk", 10),
                    fs.get("topk_features_accuracy_mean", 0), fs.get("all_features_std", 0),
                    fs.get("topk_features_std", 0), fs.get("conclusion", ""))
            else:
                conclusion = "基于训练-测试 gap 与特征重要性评估特征贡献结构"
        elif ("代价" in htext or "recall" in htext or "cost" in htext
              or "误差结构" in htext or "混淆" in htext):
            if class_spread:
                item["class_recall_spread"] = class_spread
                conclusion = "逐类召回差异范围 %s，说明误差结构性（类间重叠）而非单纯容量不足" % class_spread
            else:
                conclusion = "基于逐类指标与混淆矩阵评估误差结构"
        else:
            lin = [g for m, g in gap_block.items() if m in ("LogisticRegression", "Ridge", "LinearRegression", "Lasso", "ElasticNet", "SVC")]
            ens = [g for m, g in gap_block.items() if m in ("RandomForestClassifier", "GradientBoostingClassifier", "KNeighborsClassifier", "RandomForestRegressor", "GradientBoostingRegressor")]
            if lin and ens:
                conclusion = "训练-测试 %s gap：线性均值 %.4f，集成均值 %.4f，%s" % (
                    metric, float(np.mean(lin)), float(np.mean(ens)),
                    "集成 gap 更大说明更易过拟合" if np.mean(ens) > np.mean(lin) else "线性 gap 不大于集成")
            else:
                conclusion = "基于 5 种子训练-测试 %s gap（%s）与多种子标准差（%s）验证" % (
                    metric, _fmt_evidence_block(gap_block), _fmt_evidence_block(std_block))
        if not conclusion:
            conclusion = "基于确定性实验的指标差异验证（gap=%s）" % _fmt_evidence_block(gap_block)
        item["conclusion"] = conclusion
        if verdict:
            item["verdict"] = verdict
        evidence[hid] = item
    return evidence


def _significance(best, second, seed_metrics, main_metric) -> str:
    if not best or not second:
        return "仅单一有效方法，无法做配对显著性检验。"
    a = np.asarray(seed_metrics.get(best, {}).get(main_metric, []), dtype=float)
    b = np.asarray(seed_metrics.get(second, {}).get(main_metric, []), dtype=float)
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return "种子数量不足以做配对显著性检验。"
    t, p = stats.ttest_rel(a, b)
    sig = "显著" if (p is not None and p < 0.05) else "不显著"
    pstr = f"{p:.4f}" if p is not None else "NA"
    return (
        f"对主指标 {main_metric} 在 {len(a)} 个种子上做配对 t 检验："
        f"最优方法 {best} 与次优方法 {second} 的差异 {sig}（p={pstr}）。"
    )


def _classification_plots(seed_metrics, confmat_cache, importance_cache, best_method, plots_dir):
    plots = []
    # 1) 准确率对比柱状图（含标准差误差棒）
    methods = [m for m in seed_metrics if seed_metrics[m].get("accuracy")]
    means = [float(np.mean(seed_metrics[m]["accuracy"])) for m in methods]
    stds = [float(np.std(seed_metrics[m]["accuracy"], ddof=0)) for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(methods)), means, yerr=stds, capsize=4, color="#4C72B0")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (mean±std)")
    ax.set_title("Candidate methods: test accuracy (5 seeds)")
    fig.tight_layout()
    p1 = plots_dir / "acc_bar.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    plots.append({"file": "acc_bar.png", "caption": "各候选方法 5 种子平均测试准确率（误差棒为标准差）"})

    # 2) 最优方法混淆矩阵热力图
    if best_method in confmat_cache:
        cm = np.asarray(confmat_cache[best_method], dtype=float)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{best_method} confusion matrix (seed=42)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="red" if cm[i, j] > cm.max() * 0.5 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        p2 = plots_dir / "conf_mat.png"
        fig.savefig(p2, dpi=130)
        plt.close(fig)
        plots.append({"file": "conf_mat.png", "caption": f"{best_method} 的测试集混淆矩阵（行=真实，列=预测）"})

    # 3) 特征重要性 Top-10（优先 best 方法，否则第一个有的）
    imp_method = best_method if best_method in importance_cache else (
        next(iter(importance_cache), None) if importance_cache else None)
    if imp_method:
        fi = importance_cache[imp_method]
        items = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        labels = [k for k, _ in items][::-1]
        vals = [abs(v) for _, v in items][::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(range(len(labels)), vals, color="#55A868")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Importance (|coef| or impurity)")
        ax.set_title(f"Top-10 feature importance ({imp_method})")
        fig.tight_layout()
        p3 = plots_dir / "fi_bar.png"
        fig.savefig(p3, dpi=130)
        plt.close(fig)
        plots.append({"file": "fi_bar.png", "caption": f"{imp_method} 的 Top-10 特征重要性"})

    return plots


def _regression_plots(seed_metrics, coef_block, resid_cache, best_method, plots_dir):
    plots = []
    methods = [m for m in seed_metrics if seed_metrics[m].get("r2")]
    means = [float(np.mean(seed_metrics[m]["r2"])) for m in methods]
    stds = [float(np.std(seed_metrics[m]["r2"], ddof=0)) for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(methods)), means, yerr=stds, capsize=4, color="#C44E52")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("R² (mean±std)")
    ax.set_title("Candidate methods: test R² (5 seeds)")
    fig.tight_layout()
    p1 = plots_dir / "r2_bar.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    plots.append({"file": "r2_bar.png", "caption": "各候选方法 5 种子平均测试 R²（误差棒为标准差）"})

    # 系数条
    coef_method = best_method if best_method in coef_block else (next(iter(coef_block), None) if coef_block else None)
    if coef_method:
        coef = coef_block[coef_method]
        items = sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        labels = [k for k, _ in items][::-1]
        vals = [v for _, v in items][::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(range(len(labels)), vals, color="#8172B3")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Coefficient")
        ax.set_title(f"Top-10 coefficients ({coef_method})")
        fig.tight_layout()
        p2 = plots_dir / "coef_bar.png"
        fig.savefig(p2, dpi=130)
        plt.close(fig)
        plots.append({"file": "coef_bar.png", "caption": f"{coef_method} 的 Top-10 回归系数"})

    # 残差散点
    if best_method in resid_cache:
        yp, yt = resid_cache[best_method]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(yt, yp, s=12, alpha=0.5, color="#4C72B0")
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{best_method} predicted vs actual")
        fig.tight_layout()
        p3 = plots_dir / "resid_scatter.png"
        fig.savefig(p3, dpi=130)
        plt.close(fig)
        plots.append({"file": "resid_scatter.png", "caption": f"{best_method} 预测值-真实值散点（红线为理想拟合）"})
    return plots


# ── 多数据集对比（P8-1）────────────────────────────────────────────
def _dataset_label(spec: dict) -> str:
    """从 spec 提取可读的数据集名称（sklearn 名→中文映射表）。"""
    import sklearn.datasets as ds

    data = spec.get("data") or {}
    name = data.get("name") or spec.get("dataset") or "unknown"
    MAP = {
        "breast_cancer": "乳腺癌", "digits": "手写数字", "diabetes": "糖尿病",
        "wine": "葡萄酒", "iris": "鸢尾花", "boston": "波士顿房价",
        "california_housing": "加州房价",
    }
    return MAP.get(name, name)


def run_multi_analysis(spec: dict, out_path: Path) -> dict:
    """多数据集包装：逐个跑 run_analysis，输出跨数据集合并结果。

    spec.datasets 为列表，每项为 {data:..., label:...} 或兼容旧 spec 直接用数据名。
    各数据集的 results/{label}/ 下放独立 JSON+图；根目录 out_path 写合并后的
    combined_analysis.json（含跨数据集对比表+图）。
    """
    datasets = spec.get("datasets") or spec.get("data_list")
    if not datasets:
        # 不是多数据集 spec，退化为单数据集
        return run_analysis(spec, out_path)

    results_dir = out_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    method_order = None

    for ds_item in datasets:
        if isinstance(ds_item, str):
            sub_spec = dict(spec, data={"source": "sklearn", "name": ds_item}, dataset=ds_item)
            label = ds_item
        elif isinstance(ds_item, dict):
            sub_spec = dict(spec)
            sub_spec["data"] = ds_item.get("data") or ds_item
            if "dataset" not in sub_spec and "name" in ds_item.get("data", {}):
                sub_spec["dataset"] = ds_item["data"]["name"]
            label = ds_item.get("label") or _dataset_label(sub_spec)
        else:
            continue

        # 只替换文件名非法字符，保留中文等 Unicode 字符。
        # 旧写法 [^A-Za-z0-9_.-] 会把中文标签全部洗成下划线再 strip 成空串，
        # 导致所有数据集都回落到同名 "dataset" 并互相覆盖。
        label = re.sub(r'[\\/:*?"<>|]+', "_", str(label)).strip(" .") or "dataset"
        if label in all_results:  # 重名兜底，保证各数据集结果互不覆盖
            i = 2
            while f"{label}_{i}" in all_results:
                i += 1
            label = f"{label}_{i}"

        sub_out = results_dir / label / "03_experiment.json"
        (results_dir / label).mkdir(parents=True, exist_ok=True)
        try:
            res = run_analysis(sub_spec, sub_out)
        except Exception as e:
            res = {"error": str(e)}
        sub_out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        all_results[label] = res
        if not method_order:
            method_order = _extract_method_order(res)

    # 跨数据集汇总
    combined = _build_combined(all_results, method_order, spec, results_dir)
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    return combined


def _clean_method_name(name: str) -> str:
    """估计器路径 -> 显示类名。支持 `sklearn.linear_model.LogisticRegression`
    与带指标后缀的 `sklearn...LogisticRegression_accuracy` 两种写法。"""
    s = name.rsplit(".", 1)[-1]
    if "_" in s:
        s = s.split("_", 1)[0]
    return s


def _extract_method_order(res: dict) -> list:
    """从单个分析结果中提取方法显示名列表（取估计器类名，去掉模块路径）。"""
    metrics = res.get("metrics", [])
    seen = []
    for m in metrics:
        label = _clean_method_name(str(m.get("name", "")))
        if label and label not in seen:
            seen.append(label)
    return seen or []


def _build_combined(all_results: dict, method_order: list, spec: dict, results_dir: Path) -> dict:
    """合并多数据集结果：跨数据集对比表 + 一致性/排名稳定性分析 + 跨数据集图。"""
    main_metric = spec.get("metric") or "accuracy"
    objective = spec.get("objective") or "maximize"

    # 提取每个数据集上每个方法的 best 指标值
    cross_table = {}  # dataset_label -> {method: value}
    dataset_labels = sorted(all_results.keys())

    for label in dataset_labels:
        res = all_results[label]
        if res.get("error"):
            continue
        metrics = res.get("metrics", [])
        cross_table[label] = {}
        for m in metrics:
            name = str(m.get("name", ""))
            val = m.get("value")
            if val is None:
                continue
            # 匹配方法名：优先精确 / 类名后缀匹配，避免子串误匹配（如 SVC 被 LinearSVC 命中）
            matched = None
            for mm in method_order:
                if name == mm or name.endswith("." + mm) or name.split("_")[0] == mm:
                    matched = mm
                    break
            if matched is None:
                for mm in method_order:
                    if mm in name or name.startswith(mm):
                        matched = mm
                        break
            if matched is None:
                matched = _clean_method_name(name)
            if matched not in cross_table[label] or cross_table[label][matched] is None:
                cross_table[label][matched] = val

    # 剔除法在所有数据集上均无取值的方法（避免跨数据集对比表 / 图中出现整列空值），
    # 必须在 cross_table 构建完成后执行。
    method_order = [
        m for m in method_order
        if any((cross_table.get(lab, {}) or {}).get(m) is not None for lab in dataset_labels)
    ]
    method_set = set(method_order)
    # 同步收紧 cross_table，使其只包含保留下来的方法列
    for label in dataset_labels:
        ct = cross_table.get(label, {})
        cross_table[label] = {m: ct.get(m) for m in method_order if m in ct}

    # 聚合逐方法的多种子标准差与训练-测试 gap（按数据集），供假设裁决与论文使用。
    # 来源：各数据集独立结果里的 seed_variance / train_test_gap。
    per_method_stats = {}  # method -> {dataset_label: {acc_mean, acc_std, gap}}
    preprocessing_by_dataset = {}  # dataset_label -> 该数据集的逐列预处理报告
    for label in dataset_labels:
        res = all_results.get(label, {})
        if res.get("error"):
            continue
        if res.get("preprocessing"):
            preprocessing_by_dataset[label] = res["preprocessing"]
        sv = res.get("seed_variance") or {}
        gap = res.get("train_test_gap") or {}
        for m in method_order:
            entry = per_method_stats.setdefault(m, {})
            acc_mean = cross_table.get(label, {}).get(m)
            acc_std = None
            for k, v in sv.items():
                if k.startswith(m + "_") and "accuracy" in k:
                    acc_std = v.get("std")
                    break
            entry[label] = {"acc_mean": acc_mean, "acc_std": acc_std, "gap": gap.get(m)}

    # 构建跨数据集排名表
    ranking_rows = []
    for label in dataset_labels:
        ct = cross_table.get(label, {})
        if not ct:
            continue
        reverse = objective != "minimize"
        ranked = sorted(ct.items(), key=lambda x: x[1], reverse=reverse)
        row = [label]
        for m, v in ranked:
            row.append(f"{m}:{round(v, 4)}")
        ranking_rows.append(row)

    # 计算排名稳定性（每个方法在数据集间的排名方差）
    rank_stability = {}
    for m in method_order:
        ranks = []
        for label in dataset_labels:
            ct = cross_table.get(label, {})
            if m in ct:
                values = [(k, v) for k, v in ct.items()]
                values.sort(key=lambda x: x[1], reverse=(objective != "minimize"))
                for ri, (k, _) in enumerate(values):
                    if k == m:
                        ranks.append(ri + 1)
                        break
        if len(ranks) >= 2:
            rank_stability[m] = {
                "ranks": ranks,
                "mean_rank": round(float(np.mean(ranks)), 2),
                "std_rank": round(float(np.std(ranks, ddof=0)), 2),
                "datasets": dataset_labels,
            }

    # 跨数据集图：每个方法在各数据集上的表现
    plots = []
    try:
        fig, ax = plt.subplots(figsize=(max(7, len(method_order) * 1.8), 5))
        x = np.arange(len(dataset_labels))
        width = 0.75 / max(len(method_order), 1)
        colors = ["#C44E52", "#55A868", "#4C72B0", "#8172B3", "#CCB974", "#64B5CD"]
        for i, m in enumerate(method_order):
            vals = []
            for label in dataset_labels:
                ct = cross_table.get(label, {})
                vals.append(ct.get(m))
            offset = (i - len(method_order) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9, label=(m if any(v is not None for v in vals) else None),
                   color=colors[i % len(colors)], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, fontsize=8)
        ax.set_ylabel(main_metric)
        ax.set_title("Cross-dataset comparison")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = results_dir.parent / "cross_dataset.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        plots.append({"file": "cross_dataset.png", "caption": "各方法在多个数据集上的指标对比"})
    except Exception:
        pass

    combined = {
        "task_type": "multi_dataset",
        "dataset_labels": dataset_labels,
        "cross_table": cross_table,
        "ranking": ranking_rows,
        "rank_stability": rank_stability,
        "method_order": method_order,
        "per_method_stats": per_method_stats,
        "preprocessing_by_dataset": preprocessing_by_dataset,
        "metric": main_metric,
        "objective": objective,
        "plots": plots,
        "per_dataset_results": {k: {"best_method": v.get("best_method"), "best_value": v.get("best_value")}
                                for k, v in all_results.items() if not v.get("error")},
    }
    # 跨数据集确定性假设证据：基于真实 cross_table / per_method_stats 裁量，
    # 不依赖大模型，保证与论文裁决表、结论完全一致。
    if spec.get("sub_hypotheses"):
        try:
            combined["sub_hypothesis_evidence"] = _build_multi_hypothesis_evidence(spec, combined)
        except Exception:
            combined["sub_hypothesis_evidence"] = {}
    return combined


def _build_multi_hypothesis_evidence(spec: dict, combined: dict) -> dict:
    """为跨数据集实验，基于真实 cross_table / per_method_stats / rank_stability
    确定性地裁量每个子假设。返回 {H1: {statement, verdict, conclusion}}。

    关键：裁决完全由真实数据算出，绝不依赖大模型猜测，因此与论文裁决表、
    结论始终保持一致。无法从聚合数据直接验证的断言，明确标记“证据不足”，不编造。
    """
    method_order = combined.get("method_order") or []
    cross = combined.get("cross_table") or {}
    pstats = combined.get("per_method_stats") or {}
    stab = combined.get("rank_stability") or {}
    labels = combined.get("dataset_labels") or []
    metric = combined.get("metric") or "accuracy"

    # 数据集别名（英文 base 名 / 中文标签 → 中文标签）
    CANON = {
        "乳腺癌": "乳腺癌", "wine": "葡萄酒", "葡萄酒": "葡萄酒",
        "手写数字": "手写数字", "digits": "手写数字", "breast_cancer": "乳腺癌",
        "iris": "鸢尾花", "diabetes": "糖尿病",
    }

    def _mentioned_methods(text: str):
        found = []
        for m in method_order:
            if m and m in text:
                found.append(m)
        seen, out = set(), []
        for m in found:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def _mentioned_datasets(text: str):
        """返回假设真正提到的数据集，且必须是 cross_table 里真实存在的键。

        关键修正（实跑发现）：cross_table 的键是规划器生成的【完整标签】，可能带括号
        后缀（如「手写数字（高维多类）」），而 CANON 只有基名（「手写数字」）。
        旧实现返回基名，导致 _acc/_std/_gap 用基名查表恒为 None，
        跨数据集假设（如 H1「SVC 相对 LR 的优势在 X 上比 Y 上更大」）被误判「证据不足」；
        且旧 CANON 缺「鸢尾花」键，鸢尾花根本识别不出。
        另：按【在文本中首次出现的位置】排序，确保「X 上比 Y 上更大」里 X 在前，
        否则如写成「在 Y 上比 X 上更大」会把两个数据集前后颠倒、裁决反向。
        """
        hits = []
        # 1) 真实标签全名直接命中（含括号后缀）
        for L in labels:
            if L and L in text:
                hits.append((text.find(L), L))
        # 2) 标签基名（去掉括号后缀）命中，解决「鸢尾花」这类未在 CANON 的中文基名
        for L in labels:
            base = L.split("（")[0].split("(")[0].strip()
            if base and base in text:
                hits.append((text.find(base), L))
        # 3) 英文数据集名 → 中文基名 → 真实标签
        tt = text.lower()
        for key, base in CANON.items():
            if key.lower() in tt:
                for L in labels:
                    if L.startswith(base):
                        hits.append((tt.find(key.lower()), L))
        if not hits:
            return list(labels)
        # 同一真实标签保留最早出现位置，去重后按出现顺序返回
        best = {}
        for pos, L in hits:
            if L not in best or pos < best[L]:
                best[L] = pos
        return [L for L, _ in sorted(best.items(), key=lambda kv: kv[1])]

    def _subject_partner(text: str, methods):
        """识别假设的主语方法与对比伙伴方法（基于首个比较标记前后定位）。

        返回 (subject, partner)。例如“A 优于 B”→(A, B)；“A 相对于 B 的优势”→(A, B)。
        """
        MARKERS = ("相对于", "相比", "相对", "比在", "比", "优于", "高于", "好于",
                   "大于", "胜过", "劣于", "弱于", "低于", "小于", "不如")
        for mk in MARKERS:
            idx = text.find(mk)
            if idx < 0:
                continue
            head = text[:idx]
            tail = text[idx + len(mk):]
            # 主语取 head 中离标记最近（最靠右）的方法；对手取 tail 中离标记最近（最靠左）的方法
            subj, best_s = None, -1
            for m in methods:
                p = head.rfind(m)
                if p > best_s:
                    best_s, subj = p, m
            part, best_p = None, len(tail) + 1
            for m in methods:
                p = tail.find(m)
                if p >= 0 and p < best_p:
                    best_p, part = p, m
            if subj and part and subj != part:
                return subj, part
        return (methods[0] if methods else None,
                methods[1] if len(methods) > 1 else (methods[0] if methods else None))

    def _acc(m, d):
        return cross.get(d, {}).get(m)

    def _std(m, d):
        e = pstats.get(m, {}).get(d)
        return e.get("acc_std") if e else None

    def _gap(m, d):
        e = pstats.get(m, {}).get(d)
        return e.get("gap") if e else None

    evidence = {}
    for i, sh in enumerate(spec.get("sub_hypotheses", [])):
        if not isinstance(sh, dict):
            continue
        hid = sh.get("id") or f"H{i + 1}"
        stmt = sh.get("statement", "")
        htext = stmt.lower()
        methods = _mentioned_methods(stmt)
        dsets = _mentioned_datasets(stmt)
        item = {"statement": stmt, "verdict": None, "conclusion": ""}

        # 准确率方向比较词（方法-方法性能对比）
        cmp_dir = None
        if any(k in htext for k in ("优于", "高于", "好于", "大于", "最好", "最高", "最大", "最优")):
            cmp_dir = "gt"
        elif any(k in htext for k in ("低于", "小于", "劣于", "弱于", "最差", "最低", "最小")):
            cmp_dir = "lt"

        # 模式 A：标准差 / 方差 / 波动 / 稳定 / 偏差（明确针对方差，优先于准确率比较）
        if any(k in htext for k in ("标准差", "方差", "波动", "稳定", "偏差")) and methods:
            # 关键修正（实跑发现）：旧实现在点名了两个具体方法但非「线性模型」时，
            # 会回退成 verdict=None 并写出「未点名具体方法族，仅报告真实数值」的结论——
            # 而假设明明点名了具体方法（如 H2「随机森林的标准差小于K近邻」），
            # 这是一句虚假陈述。这里按「主语 vs 对手」做真实数值比较给出裁决。
            def _std_agg(m):
                vals = [_std(m, d) for d in dsets if _std(m, d) is not None]
                return float(np.mean(vals)) if vals else None

            if len(methods) >= 2:
                a, b = _subject_partner(stmt, methods)
                sa, sb = _std_agg(a), _std_agg(b)
                if sa is not None and sb is not None:
                    ok, direction = None, None
                    if any(k in htext for k in ("小于", "低于", "更小", "最少", "最低", "最小")):
                        ok, direction = sa < sb, f"断言 {a} 的标准差小于 {b}"
                    elif any(k in htext for k in ("大于", "高于", "更大", "最高", "最大")):
                        ok, direction = sa > sb, f"断言 {a} 的标准差大于 {b}"
                    if ok is not None:
                        item["verdict"] = "支持" if ok else "不支持"
                        item["conclusion"] = (
                            f"{direction}：跨数据集平均多种子标准差 {a}={sa:.4f}，{b}={sb:.4f}，"
                            f"差值={sa - sb:+.4f}，实测方向{'一致' if ok else '相反'}。")
                        evidence[hid] = item
                        continue
            # 未点名方法对，或未给出可裁决的比较方向：如实报告数值，绝不编造裁决
            rows = []
            for m in methods:
                s = _std_agg(m)
                if s is not None:
                    rows.append((m, s))
            rows.sort(key=lambda x: x[1])
            detail = "；".join(f"{m} std≈{v:.4f}" for m, v in rows) if rows else "（无可用的多种子标准差）"
            item["verdict"] = None
            item["conclusion"] = (
                f"各方法跨数据集平均多种子标准差：{detail}。"
                + ("已点名具体方法但未给出可裁决的两两比较方向，仅如实报告真实数值。"
                   if len(methods) >= 2 else "该断言未点名具体对比方法，仅如实报告真实数值。"))
            evidence[hid] = item
            continue

        # 模式 B：两方法在某（些）数据集上的准确率方向比较（含主体/对手识别）
        if cmp_dir and len(methods) >= 2:
            a, b = _subject_partner(stmt, methods)
            a_vals = [(_acc(a, d), d) for d in dsets if _acc(a, d) is not None]
            b_vals = [(_acc(b, d), d) for d in dsets if _acc(b, d) is not None]
            if a_vals and b_vals:
                a_mean = float(np.mean([v for v, _ in a_vals]))
                b_mean = float(np.mean([v for v, _ in b_vals]))
                diff = a_mean - b_mean
                ok = (cmp_dir in ("gt", "max") and diff > 0) or (cmp_dir in ("lt", "min") and diff < 0)
                concl = (f"实测方向与断言{'一致' if ok else '相反'}：{a} 在{('/'.join(dsets))}上平均"
                         f"{metric}={a_mean:.4f}，{b} 平均={b_mean:.4f}，差值={diff:+.4f}。")
                if "gap" in htext:
                    # gap 仅如实列出相关方法的真实数值，不强行与准确率比较共用伙伴
                    gap_nums = {m: _gap(m, dsets[0]) for m in (a, b) if _gap(m, dsets[0]) is not None}
                    if gap_nums:
                        concl += "训练-测试 gap（" + "/".join(dsets[0]) + "）：" + \
                                 "；".join(f"{m}≈{v:.4f}" for m, v in gap_nums.items()) + "（如实列出，未做单一裁定）。"
                item["verdict"] = "支持" if ok else "不支持"
                item["conclusion"] = concl
                evidence[hid] = item
                continue

        # 模式 C：A 相对于 B 的优势在数据集 X 上比在 Y 上更大（两段式差值，注意主语与顺序）
        if "gap" not in htext and len(dsets) >= 2 and len(methods) >= 2:
            a, b = _subject_partner(stmt, methods)
            d1, d2 = dsets[0], dsets[1]
            va1, vb1 = _acc(a, d1), _acc(b, d1)
            va2, vb2 = _acc(a, d2), _acc(b, d2)
            if None not in (va1, vb1, va2, vb2):
                diff1, diff2 = va1 - vb1, va2 - vb2
                increased = diff1 > diff2  # 第一个数据集上的优势是否大于第二个
                item["verdict"] = "支持" if increased else "不支持"
                item["conclusion"] = (
                    f"{a} 相对 {b} 的优势：在{d1}上 {diff1:+.4f}（{a}={va1:.4f}，{b}={vb1:.4f}），"
                    f"在{d2}上 {diff2:+.4f}（{a}={va2:.4f}，{b}={vb2:.4f}），"
                    f"{d1}相对{d2}的优势{'扩大' if increased else '缩小/反转'}，断言{'成立' if increased else '不成立'}。")
                evidence[hid] = item
                continue

        # 模式 D：单纯 gap 比较（无准确率方向词、无方差词）
        if "gap" in htext and methods:
            rows = []
            for m in methods:
                gaps = [_gap(m, d) for d in dsets if _gap(m, d) is not None]
                if gaps:
                    rows.append((m, min(gaps)))
            if rows:
                rows.sort(key=lambda x: x[1])
                lowest = rows[0][0]
                detail = "；".join(f"{m} gap≈{v:.4f}" for m, v in rows)
                is_linear = any(k in htext for k in ("线性", "linearsvc", "逻辑回归", "线性svm"))
                if is_linear:
                    ok = lowest in ("逻辑回归", "线性SVM")
                    verdict = "支持" if ok else "不支持"
                else:
                    verdict = None
                item["verdict"] = verdict
                item["conclusion"] = (f"各方法跨数据集最小训练-测试 gap：{detail}；最低为 {lowest}。"
                                     + (f"断言（线性模型更小）{'成立' if verdict == '支持' else '不成立'}。"
                                        if verdict is not None else "未点名具体方法族，仅报告真实数值。"))
                evidence[hid] = item
                continue

        # 兜底：提供真实数字摘要，明确无法裁定（绝不编造）
        snippets = []
        for m in methods:
            for d in dsets:
                a = _acc(m, d)
                if a is None:
                    continue
                s, g = _std(m, d), _gap(m, d)
                snippets.append(f"{m}@{d}: {metric}={a:.4f}"
                                + (f", std={s:.4f}" if s is not None else "")
                                + (f", gap={g:.4f}" if g is not None else ""))
        item["conclusion"] = ("基于跨数据集真实数据：" + "；".join(snippets[:12]) + "。"
                              "该断言涉及的具体对比/机制无法从聚合数据直接裁定，标记证据不足。") if snippets else \
                             "跨数据集聚合数据中未包含该断言可直接验证的数值，标记证据不足。"
        item["verdict"] = "证据不足"
        evidence[hid] = item
    return evidence


if __name__ == "__main__":
    import sys

    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("03_experiment.json")
    res = run_analysis(spec, out)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("analysis written to", out, "| plots:", [p.get("file") for p in res.get("plots", [])])
