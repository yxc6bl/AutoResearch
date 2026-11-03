"""题目驱动执行：把老师给的论文题目 / 方向，确定性地转成系统可跑的 spec。

设计要点（与「选题能力」互补、且更贴近真实毕设）：
- 真实毕设是「老师给方向甚至直接给题目 → 学生照做产出」，不是「系统替你选题」。
- 本模块零 LLM 依赖：纯关键词规则，断网 / 网关抽风也能稳定产出，可复现。
- 输出的是与 run_task 完全兼容的 spec（data / task_type / candidate_methods /
  metric / proposed_method 由 run_task 注入），从而复用既有的确定性分析与论文生成。

模块分三层：
  1) METHOD_CANON  —— 中文/英文方法别名 → 真实 sklearn 估算器（分类/回归双形态）
  2) DATASET_CATALOG —— 可绑定数据集（本地 CSV + sklearn 内置），带领域关键词
  3) 解析流程        —— 题目 → 任务类型 + 方法提示 + 领域 → 匹配数据集 → 组装 spec
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


# ─────────────────────────────────────────────────────────────────────────────
# 1) 方法规范表：别名 → 估算器（分类/回归双形态）
# ─────────────────────────────────────────────────────────────────────────────
# 每条：
#   cn      中文显示名（进 candidate_methods.name）
#   aliases 题目里可能出现的中英文写法（小写匹配）
#   clf     分类估算器全路径
#   reg     回归估算器全路径（无则仅分类可用）
#   params_clf / params_reg  默认超参；"<seed>" 占位符由引擎替换为当前随机种子
#   cat     方法大类（用于研究问题与文献检索的组织）
_METHOD_DEFS = {
    "rf": {
        "cn": "随机森林",
        "aliases": ["随机森林", "random forest", "rf", "randomforest"],
        "clf": "sklearn.ensemble.RandomForestClassifier",
        "reg": "sklearn.ensemble.RandomForestRegressor",
        "params_clf": {"n_estimators": 300, "random_state": "<seed>"},
        "params_reg": {"n_estimators": 300, "random_state": "<seed>"},
        "cat": "集成方法",
    },
    "lr": {
        "cn": "逻辑回归",
        "aliases": ["逻辑回归", "logistic", "logisticregression", "逻辑斯蒂"],
        "clf": "sklearn.linear_model.LogisticRegression",
        "reg": "sklearn.linear_model.LinearRegression",
        "params_clf": {"max_iter": 3000, "random_state": "<seed>"},
        "params_reg": {},
        "cat": "线性模型",
    },
    "svm": {
        "cn": "支持向量机",
        "aliases": ["支持向量机", "support vector", "svm", "svc", "svr", "核方法", "核函数"],
        "clf": "sklearn.svm.SVC",
        "reg": "sklearn.svm.SVR",
        "params_clf": {"random_state": "<seed>"},
        "params_reg": {},
        "cat": "核方法",
    },
    "knn": {
        "cn": "K近邻",
        "aliases": ["k近邻", "k-近邻", "knn", "k nearest", "邻近", "最近邻"],
        "clf": "sklearn.neighbors.KNeighborsClassifier",
        "reg": "sklearn.neighbors.KNeighborsRegressor",
        "params_clf": {"n_neighbors": 5},
        "params_reg": {"n_neighbors": 5},
        "cat": "基于实例",
    },
    "gb": {
        "cn": "梯度提升树",
        "aliases": ["梯度提升", "gradient boosting", "xgboost", "lightgbm", "gbdt", "gbm"],
        "clf": "sklearn.ensemble.GradientBoostingClassifier",
        "reg": "sklearn.ensemble.GradientBoostingRegressor",
        "params_clf": {"random_state": "<seed>"},
        "params_reg": {"random_state": "<seed>"},
        "cat": "集成方法",
    },
    "dt": {
        "cn": "决策树",
        "aliases": ["决策树", "decision tree", "cart"],
        "clf": "sklearn.tree.DecisionTreeClassifier",
        "reg": "sklearn.tree.DecisionTreeRegressor",
        "params_clf": {"random_state": "<seed>"},
        "params_reg": {"random_state": "<seed>"},
        "cat": "基于树",
    },
    "nb": {
        "cn": "朴素贝叶斯",
        "aliases": ["朴素贝叶斯", "naive bayes", "naivebayes", "贝叶斯", "bayes"],
        "clf": "sklearn.naive_bayes.GaussianNB",
        "reg": None,
        "params_clf": {},
        "params_reg": {},
        "cat": "概率生成模型",
    },
}

# 题目里出现这些写法但目前引擎未内置后端 → 记录为「暂不支持」，不阻塞主流程
_UNSUPPORTED_METHOD_HINTS = {
    "神经网络": "neural network / mlp",
    "深度学习": "deep learning",
    "卷积": "cnn",
    "cnn": "cnn",
    "循环神经网络": "rnn",
    "rnn": "rnn",
    "lstm": "lstm",
    "transformer": "transformer",
}

# 各任务类型的默认候选方法（题目未指定方法时采用）
_DEFAULT_METHODS = {
    "classification": ["rf", "lr", "svm", "knn", "gb"],
    "regression": ["rf", "lr", "svm", "knn", "gb"],
    "clustering": ["kmeans", "agglomerative"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 2) 数据集目录：可被题目匹配的真实数据集
#    domain_kw：题目/方向里出现即视为匹配（小写匹配；中文为子串）
# ─────────────────────────────────────────────────────────────────────────────
DATASET_CATALOG = [
    {
        "key": "heart",
        "label": "UCI 心脏病数据集",
        "source": "csv",
        "path": str(DATA_DIR / "heart_disease.csv"),
        "target": "target",
        "task": "classification",
        "domain_kw": ["心脏病", "心血管", "heart", "cardio", "临床", "患病", "冠心病"],
        "lit_query": "heart disease cardiovascular risk prediction machine learning clinical features",
        "desc_cn": "心血管疾病风险预测",
        "target_cn": "是否罹患心血管疾病",
        "n": 303, "feat": 13,
    },
    {
        "key": "airfoil",
        "label": "NASA 翼型气动噪声数据集",
        "source": "csv",
        "path": str(DATA_DIR / "airfoil_noise.csv"),
        "target": "sound_pressure_db",
        "task": "regression",
        "domain_kw": ["翼型", "气动", "噪声", "aerofoil", "airfoil", "声压", "气动声学"],
        "lit_query": "airfoil aerodynamic noise prediction machine learning aeroacoustics regression",
        "desc_cn": "翼型气动噪声预测",
        "target_cn": "声压级（分贝）",
        "n": 1503, "feat": 5,
    },
    {
        "key": "iris",
        "label": "鸢尾花数据集（sklearn 内置）",
        "source": "sklearn",
        "name": "iris",
        "task": "classification",
        "domain_kw": ["鸢尾花", "iris", "花", "植物分类"],
        "lit_query": "iris flower classification machine learning benchmark",
        "desc_cn": "鸢尾花品种分类",
        "target_cn": "鸢尾花品种类别",
        "n": 150, "feat": 4,
    },
    {
        "key": "wine",
        "label": "葡萄酒数据集（sklearn 内置）",
        "source": "sklearn",
        "name": "wine",
        "task": "classification",
        "domain_kw": ["葡萄酒", "wine", "酿酒", "品种鉴别"],
        "lit_query": "wine quality classification machine learning",
        "desc_cn": "葡萄酒品种分类",
        "target_cn": "葡萄酒品种类别",
        "n": 178, "feat": 13,
    },
    {
        "key": "breast_cancer",
        "label": "乳腺癌数据集（sklearn 内置）",
        "source": "sklearn",
        "name": "load_breast_cancer",
        "task": "classification",
        "domain_kw": ["乳腺癌", "乳腺", "breast cancer", "cancer", "肿瘤", "诊断"],
        "lit_query": "breast cancer diagnosis machine learning classification",
        "desc_cn": "乳腺肿瘤良恶性诊断",
        "target_cn": "肿瘤良恶性类别",
        "n": 569, "feat": 30,
    },
    {
        "key": "digits",
        "label": "手写数字数据集（sklearn 内置）",
        "source": "sklearn",
        "name": "load_digits",
        "task": "classification",
        "domain_kw": ["手写数字", "digit", "数字识别", "字符识别"],
        "lit_query": "handwritten digit recognition machine learning classification",
        "desc_cn": "手写数字识别",
        "target_cn": "数字类别（0-9）",
        "n": 1797, "feat": 64,
    },
    {
        "key": "diabetes",
        "label": "糖尿病数据集（sklearn 内置）",
        "source": "sklearn",
        "name": "load_diabetes",
        "task": "regression",
        "domain_kw": ["糖尿病", "diabetes", "血糖", "进展"],
        "lit_query": "diabetes progression regression machine learning",
        "desc_cn": "糖尿病病情进展预测",
        "target_cn": "一年后的疾病进展量化值",
        "n": 442, "feat": 10,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 3) 解析流程
# ─────────────────────────────────────────────────────────────────────────────
def _norm(text: str) -> str:
    return (text or "").lower()


def detect_task_type(title: str, direction: str = "", matched_task: str = None) -> str:
    """从题目/方向推断任务类型。匹配到数据集时以其 task 为准（最权威）。"""
    if matched_task:
        return matched_task
    t = _norm(title + " " + direction)
    # 聚类优先（最特异）
    if any(k in t for k in ["聚类", "分群", "簇", "无监督", "k-means", "kmeans", "cluster"]):
        return "clustering"
    # 回归指示词（连续量）
    if any(k in t for k in ["回归", "估计", "预测值", "强度", "温度", "噪声", "浓度",
                            "负荷", "房价", "气压", "应变", "含量", "预测（连续）", "regression"]):
        return "regression"
    # 分类指示词（类别）
    if any(k in t for k in ["分类", "识别", "诊断", "检测", "判别", "故障", "欺诈", "信用",
                            "患病", "风险", "类别", "品种", "预测（类别）", "classification"]):
        return "classification"
    # 默认：分类（机器学习毕设最常见）
    return "classification"


def extract_method_hints(title: str, direction: str = "") -> tuple[list[str], list[str]]:
    """返回 (支持的方法 key 列表, 暂不支持的方法名列表)。"""
    t = _norm(title + " " + direction)
    supported, unsupported = [], []
    for key, d in _METHOD_DEFS.items():
        if any(a in t for a in d["aliases"]):
            if key not in supported:
                supported.append(key)
    for hint, tag in _UNSUPPORTED_METHOD_HINTS.items():
        if hint in t and tag not in unsupported:
            unsupported.append(tag)
    return supported, unsupported


def match_dataset(title: str, direction: str = "", task_type: str = None) -> dict | None:
    """按领域关键词重叠匹配最合适的数据集；命中 0 个关键词返回 None。"""
    t = _norm(title + " " + direction)
    best, best_score = None, 0
    for ds in DATASET_CATALOG:
        score = sum(1 for kw in ds["domain_kw"] if kw.lower() in t)
        # 任务类型一致时加权，避免「心脏病」被错配到回归集
        if task_type and ds.get("task") == task_type:
            score += 0.5
        if score > best_score:
            best, best_score = ds, score
    return best if best_score > 0 else None


def build_candidate_methods(task_type: str, method_hints: list[str]) -> list[dict]:
    """把方法 key 列表转成 run_task 兼容的 candidate_methods（含中文名/估算器/超参）。

    规则：题目指定的方法作为主方法排在最前；始终补齐基线方法，形成 ≥4 个的
    可对比模型集合（统一基准对比 + 融合集成方法贡献都依赖「多个候选方法」）。
    聚类走单独的内置方法名（引擎聚类分支直接识别 kmeans/agglomerative）。
    """
    if task_type == "clustering":
        return [{"name": "K均值", "estimator": "kmeans"},
                {"name": "层次聚类", "estimator": "agglomerative"}]
    keys: list[str] = list(method_hints) if method_hints else []
    # 补齐基线，封顶 5 个，保证有对照且能构造融合集成
    for k in _DEFAULT_METHODS.get(task_type, []):
        if len(keys) >= 5:
            break
        if k not in keys:
            keys.append(k)
    out = []
    seen_est: set[str] = set()
    is_reg = (task_type == "regression")
    for k in keys:
        d = _METHOD_DEFS.get(k)
        if not d:
            continue
        est = d["reg"] if (is_reg and d["reg"]) else d["clf"]
        if est in seen_est:  # 按估算器去重（分类/回归同 key 不会同时出现）
            continue
        seen_est.add(est)
        params = (d["params_reg"] if (is_reg and d["reg"]) else d["params_clf"]).copy()
        out.append({"name": d["cn"], "estimator": est, "params": params})
    return out


def _data_binding(ds: dict) -> dict:
    if ds["source"] == "csv":
        return {"source": "csv", "path": ds["path"], "target": ds["target"]}
    if ds["source"] == "sklearn":
        return {"source": "sklearn", "name": ds["name"]}
    raise ValueError(f"未知数据源：{ds['source']}")


def assemble_title_spec(title: str, direction: str = "", data_override: dict = None,
                        method_hints: list[str] = None) -> dict:
    """题目 → 完整 spec（proposed_method 由 run_task 注入）。

    data_override: 学生自带数据 {"source":"csv","path":...,"target":...} 或
                   {"source":"sklearn","name":...}，优先级最高。
    返回 dict：{title, task_type, metric, objective, candidate_methods, data,
               literature_query, research_question, hypothesis_statement, direction,
               _match_note}
    """
    # 1) 数据集匹配（决定 task_type 权威值）
    matched = match_dataset(title, direction) if not data_override else None
    if data_override:
        task_type = detect_task_type(title, direction)
        ds_for_meta = None
    else:
        task_type = detect_task_type(title, direction, matched["task"] if matched else None)
        ds_for_meta = matched

    # 2) 方法提示
    hints, unsupported = extract_method_hints(title, direction)
    if method_hints:  # 显式覆盖
        hints = [h for h in method_hints if h in _METHOD_DEFS]
    candidate_methods = build_candidate_methods(task_type, hints)

    # 3) 数据集绑定
    if data_override:
        data = data_override
        match_note = "使用用户自带/指定数据集"
    elif matched:
        data = _data_binding(matched)
        match_note = f"匹配本地数据集：{matched['label']}（{matched['n']}样本/{matched['feat']}特征）"
    else:
        data = None
        match_note = "未匹配到本地数据集，请用 --data/--dataset 指定，或扩充 DATASET_CATALOG"

    # 4) 指标与优化方向
    metric_map = {"classification": "accuracy", "regression": "r2", "clustering": "ari"}
    metric = metric_map.get(task_type, "accuracy")
    objective = "maximize"

    # 5) 研究问题 / 假设（确定性模板，结合数据集语义）
    desc = ds_for_meta["desc_cn"] if ds_for_meta else "该给定任务"
    target_cn = ds_for_meta["target_cn"] if ds_for_meta else "目标变量"
    methods_cn = "、".join(m["name"] for m in candidate_methods) or "多种机器学习模型"
    research_question = (
        f"在{desc}任务中，不同归纳偏置的机器学习模型（{methods_cn}）"
        f"在{('准确率' if task_type=='classification' else 'R²' if task_type=='regression' else '聚类一致性')}"
        f"上是否存在显著差异？哪种模型对{target_cn}的预测更稳健？"
    )
    hypothesis_statement = (
        f"假设：集成类方法（如随机森林、梯度提升树）能通过降低方差提升预测稳定性，"
        f"其表现优于单一线性或核方法，且该结论在多种子评估下保持稳健。"
    )

    lit_query = (ds_for_meta["lit_query"] if ds_for_meta
                 else (direction or title))

    spec = {
        "title": title,
        "task_type": task_type,
        "metric": metric,
        "objective": objective,
        "candidate_methods": candidate_methods,
        "data": data,
        "literature_query": lit_query,
        "research_question": research_question,
        "hypothesis_statement": hypothesis_statement,
        "direction": direction,
        "_match_note": match_note,
        "_unsupported_methods": unsupported,
    }
    return spec


def parse_title(title: str, direction: str = "") -> dict:
    """供调试/预览：返回解析出的「意图」而不组装完整 spec。"""
    matched = match_dataset(title, direction)
    task_type = detect_task_type(title, direction, matched["task"] if matched else None)
    hints, unsupported = extract_method_hints(title, direction)
    return {
        "title": title,
        "task_type": task_type,
        "matched_dataset": matched["label"] if matched else None,
        "method_hints": [(_METHOD_DEFS[k]["cn"] if k in _METHOD_DEFS else k) for k in hints],
        "unsupported_methods": unsupported,
    }


if __name__ == "__main__":
    import json
    samples = [
        "基于随机森林的心脏病预测研究",
        "基于梯度提升的翼型气动噪声预测研究",
        "基于支持向量机的鸢尾花分类研究",
        "基于卷积神经网络的乳腺肿瘤诊断研究",
        "葡萄酒品种鉴别的机器学习方法对比",
    ]
    for s in samples:
        print(json.dumps(parse_title(s), ensure_ascii=False))
