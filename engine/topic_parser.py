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
    "mlp": {
        "cn": "多层感知机（神经网络 / 深度学习）",
        # 别名覆盖题目里常见的「神经网络 / 深度学习 / 卷积 / CNN / RNN / LSTM / Transformer」
        # 等写法：当前引擎可用的深度学习后端即 sklearn 多层感知机（MLP），对这些写法
        # 一律按 MLP 接入，使「基于深度学习的 X」类题目真正可跑，而非标记为「不支持」。
        # 注：CNN/RNN/LSTM 是更专门的架构，若未来接入 torch 可在此扩展，但 MLP 作为
        # 通用神经网络基线，已能支撑绝大多数表格 / 图像扁平化 / 文本 TF-IDF 场景。
        "aliases": ["多层感知机", "神经网络", "深度学习", "mlp", "multilayer perceptron",
                    "neural network", "deep learning", "全连接网络", "fcnn",
                    "卷积", "cnn", "卷积神经网络", "循环神经网络", "rnn", "lstm",
                    "transformer", "注意力", "attention"],
        "clf": "sklearn.neural_network.MLPClassifier",
        "reg": "sklearn.neural_network.MLPRegressor",
        "params_clf": {"hidden_layer_sizes": [64, 32], "max_iter": 500,
                       "early_stopping": True, "random_state": "<seed>"},
        "params_reg": {"hidden_layer_sizes": [64, 32], "max_iter": 500,
                       "early_stopping": True, "random_state": "<seed>"},
        "cat": "神经网络",
    },
}

# 题目里出现这些写法但目前引擎未内置后端 → 记录为「暂不支持」，不阻塞主流程。
# 现仅保留本引擎确实无法承载的写法（如强化学习、图神经网络、生成对抗网络等）。
# 注意：「深度学习 / 神经网络 / CNN / RNN / LSTM / Transformer」已统一由 mlp 接入，
# 不再列为不支持，从而让「基于深度学习的 X」类题目可跑。
_UNSUPPORTED_METHOD_HINTS = {
    "强化学习": "reinforcement learning / rl",
    "图神经网络": "graph neural network / gnn",
    "生成对抗网络": "gan",
    "gan": "gan",
}

# 各任务类型的默认候选方法（题目未指定方法时采用）。
# 含 mlp（神经网络/深度学习基线），使每篇论文默认就有 6 个可对比模型，
# 含一个真实深度学习基线；题目点名的方法始终排在最前。
_DEFAULT_METHODS = {
    "classification": ["rf", "mlp", "lr", "svm", "knn", "gb"],
    "regression": ["rf", "mlp", "lr", "svm", "knn", "gb"],
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
    # ── 扩展数据集：覆盖更多真实毕设高频主题（房价 / 图像 / 文本 / 信用 / 营销） ──
    {
        "key": "california_housing",
        "label": "加州房价数据集（sklearn 内置）",
        "source": "sklearn",
        "name": "fetch_california_housing",
        "task": "regression",
        "modality": "tabular",
        "domain_kw": ["房价", "housing", "california", "房地产", "价格预测",
                      "回归预测", "地产"],
        "lit_query": "california housing price prediction regression machine learning",
        "desc_cn": "加州房价预测",
        "target_cn": "街区房价中位数（万美元）",
        "n": 20640, "feat": 8,
    },
    {
        "key": "olivetti_faces",
        "label": "Olivetti 人脸数据集（sklearn 内置，像素扁平化）",
        "source": "sklearn",
        "name": "fetch_olivetti_faces",
        "task": "classification",
        "modality": "image",
        "domain_kw": ["人脸", "面部", "人脸识别", "face", "faces", "olivetti",
                      "图像分类", "图像识别", "像素", "面部识别"],
        "lit_query": "face recognition classification machine learning olivetti eigenfaces",
        "desc_cn": "人脸身份识别",
        "target_cn": "人物类别（40 类）",
        "n": 400, "feat": 4096,
    },
    {
        "key": "newsgroups",
        "label": "20 类新闻组文本数据集（sklearn 内置，TF-IDF 管道）",
        "source": "text",
        "name": "fetch_20newsgroups",
        "task": "classification",
        "modality": "text",
        "domain_kw": ["文本", "新闻", "情感", "语义", "自然语言", "文档分类",
                      "主题分类", "text", "newsgroup", "nlp", "语料", "文本分类"],
        "lit_query": "text classification 20 newsgroups TF-IDF machine learning",
        "desc_cn": "文本主题分类",
        "target_cn": "新闻组主题类别（20 类）",
        "n": 18000, "feat": "tfidf",
    },
    {
        "key": "text_sample",
        "label": "本地文本样例数据集（离线可跑，TF-IDF 管道）",
        "source": "text",
        "path": "data/text_sample.csv",
        "text_col": "text",
        "target": "label",
        "task": "classification",
        "modality": "text",
        "domain_kw": ["文本", "文档", "语料", "主题分类", "文本分类", "情感", "语义"],
        "lit_query": "text classification TF-IDF local sample machine learning",
        "desc_cn": "文本主题分类（本地样例）",
        "target_cn": "文档主题类别",
        "n": 105, "feat": "tfidf",
    },
    # ── OpenML 经典数据集（运行时联网拉取并缓存；直接对应信用/营销/质量/分类等高频题） ──
    {
        "key": "openml_adult",
        "label": "Adult 收入预测数据集（OpenML 1590）",
        "source": "openml",
        "dataset_id": 1590,
        "task": "classification",
        "modality": "tabular",
        "domain_kw": ["收入", "信用", "信用评分", "成人", "adult", "income",
                      "人口普查", "客户分群", "分群", "薪资"],
        "lit_query": "adult income prediction classification census machine learning",
        "desc_cn": "居民收入是否超 5 万美元预测",
        "target_cn": "收入档位（>50K / <=50K）",
    },
    {
        "key": "openml_bank",
        "label": "Bank Marketing 银行营销数据集（OpenML 1461）",
        "source": "openml",
        "dataset_id": 1461,
        "task": "classification",
        "modality": "tabular",
        "domain_kw": ["银行", "营销", "bank", "营销响应", "客户", "认购", "理财",
                      "电话营销", "定期存款"],
        "lit_query": "bank marketing term deposit prediction classification",
        "desc_cn": "银行定期存款认购预测",
        "target_cn": "是否认购定期存款",
    },
    {
        "key": "openml_wine_quality",
        "label": "Wine Quality 葡萄酒质量数据集（OpenML 40498）",
        "source": "openml",
        "dataset_id": 40498,
        "task": "regression",
        "modality": "tabular",
        "domain_kw": ["葡萄酒", "wine", "质量", "quality", "口感", "酿酒", "红酒"],
        "lit_query": "wine quality prediction regression machine learning",
        "desc_cn": "葡萄酒质量评分预测",
        "target_cn": "质量评分（连续）",
    },
    {
        "key": "openml_mushroom",
        "label": "Mushroom 蘑菇可食性数据集（OpenML 900）",
        "source": "openml",
        "dataset_id": 900,
        "task": "classification",
        "modality": "tabular",
        "domain_kw": ["蘑菇", "mushroom", "毒性", "可食", "真菌", "毒菌"],
        "lit_query": "mushroom edibility classification machine learning",
        "desc_cn": "蘑菇可食性判别",
        "target_cn": "可食/有毒类别",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 3) 解析流程
# ─────────────────────────────────────────────────────────────────────────────
def _norm(text: str) -> str:
    return (text or "").lower()


def detect_contribution_intent(title: str, direction: str = "") -> bool:
    """判断选题是否需要「本文方法贡献」（而非忠实复现/对比）。

    关键修复（用户反馈）：此前系统无论题目如何，都硬塞一个「融合集成方法」当贡献，
    导致「基于随机森林」这类题目被强加集成贡献而跑题。本函数让贡献是否注入
    由题目语义决定：

      - 含明确改进/融合/提出类词（改进、优化、增强、融合、集成、新型、提出、轻量、快速、
        improve、novel、propose、fusion、ensemble…）→ 需要方法贡献（True）；
      - 仅点名某方法做「基于/应用/研究/对比/识别/预测/分析」（忠实复现）→ 不需要（False），
        此时论文应忠实围绕该方法来做严谨的实证对比，而非另起炉灶造一个集成；
      - 既无改进词也无忠实词（极泛题目）→ 默认给系统一个轻量集成贡献（True），
        避免产出纯空对比、毫无增量。

    返回 True 表示 run_task 应注入 proposed_method（融合集成）；False 表示忠实模式，
    论文围绕题目点名的方法展开，不强行套贡献模板。
    """
    t = _norm(title + " " + direction)
    CONTRIB = ["改进", "优化", "增强", "提升", "融合", "集成", "新型", "新算法", "新模型",
               "提出", "轻量", "快速", "改进型", "improve", "enhanced", "novel",
               "propose", "fusion", "ensemble", "lightweight"]
    FAITHFUL = ["基于", "应用", "研究", "分类", "识别", "诊断", "检测", "预测", "对比",
                "比较", "实证", "分析", "调查", "based on", "application", "survey",
                "comparison", "empirical"]
    if any(k in t for k in CONTRIB):
        return True
    if any(k in t for k in FAITHFUL):
        return False
    return True


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
    """按领域关键词重叠匹配最合适的数据集；命中 0 个关键词返回 None。

    加权规则：
      - 领域关键词命中 +1/个；
      - 任务类型一致 +0.5；
      - 模态一致（题目提「图像/人脸」→ image 集；提「文本/语义」→ text 集）+1，
        从源头避免「图像题被错配到表格集」这类误匹配。
    """
    t = _norm(title + " " + direction)
    t_img = any(k in t for k in ["图像", "人脸", "面部", "像素", "img", "image", "face"])
    t_text = any(k in t for k in ["文本", "语义", "自然语言", "nlp", "语料", "文档", "新闻"])
    best, best_score = None, 0
    for ds in DATASET_CATALOG:
        score = sum(1 for kw in ds["domain_kw"] if kw.lower() in t)
        # 任务类型一致时加权，避免「心脏病」被错配到回归集
        if task_type and ds.get("task") == task_type:
            score += 0.5
        modality = ds.get("modality", "tabular")
        if t_img and modality == "image":
            score += 1.0
        if t_text and modality == "text":
            score += 1.0
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
    # 补齐基线，封顶 6 个，保证有对照且能构造融合集成
    for k in _DEFAULT_METHODS.get(task_type, []):
        if len(keys) >= 6:
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
    # 统一附带可读元信息（label/desc/target/modality），供论文正文明确定位数据集，
    # 避免新数据源在正文里退化成「指定数据集」这类占位词。
    meta = {"label": ds.get("label"), "desc_cn": ds.get("desc_cn"),
            "target_cn": ds.get("target_cn"), "modality": ds.get("modality", "tabular")}
    if ds["source"] == "csv":
        return {"source": "csv", "path": ds["path"], "target": ds["target"], **meta}
    if ds["source"] == "sklearn":
        return {"source": "sklearn", "name": ds["name"], **meta}
    if ds["source"] == "text":
        return {"source": "text", "name": ds.get("name"), **meta}
    if ds["source"] == "openml":
        return {"source": "openml", "dataset_id": ds["dataset_id"],
                "version": ds.get("version"), **meta}
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
        _n, _f = matched.get("n"), matched.get("feat")
        _scale = (f"{_n}样本/{_f}特征" if (_n and _f)
                  else f"OpenML data_id={matched.get('dataset_id')}" if matched.get("source") == "openml"
                  else "规模按实际加载")
        match_note = f"匹配本地数据集：{matched['label']}（{_scale}）"
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
        # 忠实选题：是否需要方法贡献。由题目语义决定，run_task 据此决定是否注入融合集成。
        # True=需要（题目要改进/融合/提出）；False=忠实复现（题目基于某方法做研究/对比）。
        "_needs_contribution": detect_contribution_intent(title, direction),
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
        "matched_modality": matched.get("modality", "tabular") if matched else None,
        "method_hints": [(_METHOD_DEFS[k]["cn"] if k in _METHOD_DEFS else k) for k in hints],
        "unsupported_methods": unsupported,
        "needs_contribution": detect_contribution_intent(title, direction),
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
