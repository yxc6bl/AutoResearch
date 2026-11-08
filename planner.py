"""LLM Planner（AutoResearch 子模块）——「喂 spec 的 agent」。

职责：把一句自然语言的研究主题，变成一份**合法、引擎可直接跑**的任务 spec。
这是把系统从「人写 spec」升级为「agent 写 spec、引擎跑」的闭环起点。

- 有 API Key 时：调 DeepSeek 产出数据集 / 候选方法 / 研究问题 / 子假设 / 指标，
  并落到一个可被 analysis_engine.run_analysis 直接消费的 JSON。
- 无 Key 时：降级为基于经典数据集的「模板 spec」，保证流水线永不卡死。

入口：
    from planner import plan_spec, llm_client
    client, model, extra = llm_client()
    spec = plan_spec("比较随机森林与逻辑回归在糖尿病预测上的表现", client, model, extra)

命令行（调试用）：
    python planner.py "你的研究主题"
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ── 离线可用数据集白名单（无需联网） ─────────────────────────────
OFFLINE_CLASSIFICATION = ["breast_cancer", "digits", "wine", "iris"]
OFFLINE_REGRESSION = ["diabetes"]

# 常见字符串方法名 -> 完整 sklearn import 路径（兼容 LLM 偷懒写短名）
METHOD_ALIAS = {
    "logisticregression": "sklearn.linear_model.LogisticRegression",
    "randomforestclassifier": "sklearn.ensemble.RandomForestClassifier",
    "randomforestregressor": "sklearn.ensemble.RandomForestRegressor",
    "svc": "sklearn.svm.SVC",
    "svr": "sklearn.svm.SVR",
    "kneighborsclassifier": "sklearn.neighbors.KNeighborsClassifier",
    "kneighborsregressor": "sklearn.neighbors.KNeighborsRegressor",
    "gradientboostingclassifier": "sklearn.ensemble.GradientBoostingClassifier",
    "gradientboostingregressor": "sklearn.ensemble.GradientBoostingRegressor",
    "linearregression": "sklearn.linear_model.LinearRegression",
    "ridge": "sklearn.linear_model.Ridge",
    "lasso": "sklearn.linear_model.Lasso",
    "decisiontreeclassifier": "sklearn.tree.DecisionTreeClassifier",
    "decisiontreeregressor": "sklearn.tree.DecisionTreeRegressor",
    "xgbclassifier": "xgboost.XGBClassifier",
    "xgbregressor": "xgboost.XGBRegressor",
    "lgbmclassifier": "lightgbm.LGBMClassifier",
    "lgbmregressor": "lightgbm.LGBMRegressor",
}


def _load_dotenv() -> dict:
    env = dict(os.environ)
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def llm_client():
    """返回 (client, model, extra)；无 Key 时返回 (None, None, None)。"""
    env = _load_dotenv()
    key = env.get("OPENAI_API_KEY")
    if not key or key.startswith("在此"):
        return None, None, None
    try:
        import openai
    except ImportError:
        return None, None, None
    base = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = env.get("OPENAI_MODEL", "deepseek-v4-pro")
    try:
        extra = json.loads(env.get("LLM_EXTRA_BODY", "{}") or "{}")
    except Exception:
        extra = {}
    try:
        client = openai.OpenAI(api_key=key, base_url=base, timeout=60, max_retries=0)
    except Exception:
        return None, None, None
    return client, model, extra


def _llm_json(client, model, extra, prompt, temperature=0.2):
    """调用 LLM 并尽量解析出 JSON。"""
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        extra_body=extra or None,
        timeout=60,
    )
    text = r.choices[0].message.content or ""
    return _extract_json(text)


def _extract_json(text: str):
    """从 LLM 输出里鲁棒地提取 JSON（兼容 ```json 围栏、前后多余文本）。"""
    if not text:
        return None
    # 优先取 ```json ... ``` 围栏
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 再取第一个 { 到最后一个 }
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(text[a : b + 1])
        except Exception:
            pass
    return None


_SYSTEM = """你是一个机器学习研究规划器。用户给一句研究主题，你要输出一份
JSON 形式的「任务规格（task spec）」，供一个确定性实验引擎直接运行。

可用离线数据集（只能从这些里选，且必须与任务类型匹配）：
- 分类：breast_cancer（乳腺癌良恶性）、digits（手写数字）、wine（葡萄酒品种）、iris（鸢尾花）
- 回归：diabetes（糖尿病进展）

候选方法必须从下列 sklearn 估计器里选 4~5 个，且至少覆盖 3 种不同归纳偏置族
（如：线性模型 / 树集成 / 核方法 / 近邻），避免全部来自同一族；用 dict 写法，
需随机种子的方法把 random_state 写成字符串 "<seed>"（引擎会替换）：
- 分类：sklearn.linear_model.LogisticRegression（params:{"max_iter":3000,"random_state":"<seed>"}）
         sklearn.ensemble.RandomForestClassifier（params:{"n_estimators":300,"random_state":"<seed>"}）
         sklearn.svm.SVC（params:{"random_state":"<seed>"}）
         sklearn.neighbors.KNeighborsClassifier（params:{"n_neighbors":5}）
         sklearn.ensemble.GradientBoostingClassifier（params:{"random_state":"<seed>"}）
- 回归：sklearn.linear_model.LinearRegression
         sklearn.linear_model.Ridge（params:{"random_state":"<seed>"}）
         sklearn.linear_model.Lasso（params:{"random_state":"<seed>"}）
         sklearn.ensemble.RandomForestRegressor（params:{"n_estimators":300,"random_state":"<seed>"}）
         sklearn.svm.SVR

指标约定：分类用 "accuracy"（objective maximize）或 "f1_macro"；回归用 "r2"（maximize）或 "mse"（minimize）。

★★★ 数据集范围决定：单数据集 vs 多数据集（务必先做这一步）★★★
先判断研究主题是否【明确要求跨多个数据集比较】（出现「跨数据集」「多个数据集」「不同数据集上」「dataset-level 稳定性」「排名一致性/迁移」等意图）。
情形一：要求跨数据集比较 → 你必须同时给出 "data"（第一个数据集）和 "datasets"（2~3 个数据集的列表），
        全部数据集必须与 task_type 匹配（分类只能选 breast_cancer/digits/wine/iris；回归只能选 diabetes），
        并挑选特征结构差异明显的组合（例如低维低样本 + 高维多类），这样跨数据集比较才有信息量。
        "task_type" 仍填 classification/regression（引擎会依据你给的 datasets 自动进入多数据集模式）。
        此时 research_question / 子假设可以正当谈论「跨数据集排名一致性/稳定性」。
情形二：只研究一个数据集（未要求跨数据集） → 只给 "data"，不要给 "datasets"。
        ★ 关键：此时研究只在单一数据集上展开，research_question 与 hypothesis 必须收敛到「在该数据集上，
        比较若干方法的…」，严禁出现「跨数据集迁移」「不同特征结构（如 A/B/C 三类数据）的表格数据上」
        「哪种归纳偏置在跨数据集迁移时…」这类研究者实际上并没有做的对比承诺——一旦写了，
        论文就会出现「声称跨数据集、实际只跑了一个数据集」的虚假陈述。
这一条必须同时贯彻到 research_question 的措辞里。

可选进阶方法（需运行环境安装对应库，未安装时确定性引擎会自动跳过该方法并继续）：
- 分类：xgboost.XGBClassifier（params:{"n_estimators":300,"random_state":"<seed>"}）、lightgbm.LGBMClassifier（params:{"n_estimators":300,"random_state":"<seed>"}）
- 回归：xgboost.XGBRegressor（params:{"n_estimators":300,"random_state":"<seed>"}）、lightgbm.LGBMRegressor（params:{"n_estimators":300,"random_state":"<seed>"}）
- 深度学习（需安装 torch）：{"name":"PyTorchMLP","estimator":"PyTorchMLP","params":{"hidden":[64,32],"epochs":60,"task":"classification"}}

子假设必须可证伪且具体到机制：每条都要点名具体方法（用上面声明的中文名）与具体机理
（如「高相关特征下，带 L2 正则的逻辑回归优于无正则版本」「树集成在该数据集的方差显著低于线性模型」），
禁止写「方法各有优势」之类的空话；expected 字段写清楚你预期哪一方胜出及原因。

★★ 子假设硬约束（违反会被下游确定性引擎判为「证据不足」，务必遵守）★★
1. 子假设里提到的「方法」只能是上面 candidate_methods 中声明过的中文名；
   严禁出现未声明的算法（如高斯朴素贝叶斯、XGBoost、LightGBM 等，除非你已把它们列入 candidate_methods）。
2. 子假设只能围绕本 spec 的 metric 字段（即你写在下方 JSON 的 metric 实际值，如 accuracy / f1_macro / r2 / mse）展开；
   禁止擅自改用其它指标（例如声明了 accuracy 却去谈「F1 宏平均方差」「AUC」等引擎没测的量）。
3. 子假设必须是「可量化比较」的句式，确定性引擎只认下面三类（照抄结构即可）：
   (A) 稳定性类：「在多种子下，<方法甲> 的主指标标准差小于 <方法乙>」；
   (B) 两两比较类：「在主指标上，<方法甲> 显著优于 <方法乙>」；
   (C) 多数据集类（仅当你这一轮用了多个数据集时）：「<方法甲> 相对 <方法乙> 的优势在数据集 X 上比数据集 Y 上更大」。
4. 禁止写「主假设已验证」「本文证明」等引擎无法核实或循环论证的表述。
遵守以上约束，子假设才能被引擎给出真实裁决，而不是一律「证据不足」。

严格只输出如下 JSON，不要任何解释，字段必须齐全：
{
  "title": "论文标题",
  "research_question": "本文要回答的一个具体机理问题",
  "task_type": "classification 或 regression",
  "data": {"source": "sklearn", "name": "从上述白名单选一个"},
  "datasets": [{"data": {"source": "sklearn", "name": "从白名单选，且与task_type匹配"}, "label": "中文数据集名"}, {"data": {...}, "label": "..."}],
  "candidate_methods": [
    {"name": "中文可读名", "estimator": "完整import路径", "params": {...}}
  ],
  "metric": "accuracy/f1_macro/r2/mse",
  "objective": "maximize/minimize",
  "sub_hypotheses": [
    {"id":"H1","statement":"假设陈述","rationale":"为什么合理","expected":"预期结果"},
    {"id":"H2", ...},
    {"id":"H3", ...}
  ],
  "analysis_plan": "一句话描述如何比较方法、看什么指标",
  "literature_query": "用于检索 arXiv 的英文关键词，3~6 个词"
}"""


def plan_spec(topic: str, client=None, model=None, extra=None, data_override: dict | None = None):
    """把研究主题转成合法 spec。client 为 None 时返回模板 spec。"""
    if client is None:
        return template_spec(topic, data_override)
    prompt = (
        _SYSTEM
        + f"\n\n研究主题：{topic}\n\n请给出上述 JSON。"
    )
    try:
        spec = _llm_json(client, model, extra, prompt)
    except Exception:
        spec = None
    if not isinstance(spec, dict):
        return template_spec(topic, data_override)
    return validate_and_fix(spec, topic, data_override)


_ITERATE_SYSTEM = """你是机器学习研究的「迭代规划器」。给定上一轮的实验规格与真实结果，
输出一份**扩宽或深化**的研究规格 JSON，使下一轮实验更有信息量。

扩宽原则：
1. 在上一轮候选方法基础上，新增 1~2 个能检验新机制的方法（仍只能从 sklearn 估计器选，
   classification: LogisticRegression/RandomForestClassifier/SVC/KNeighborsClassifier/GradientBoostingClassifier；
   regression: LinearRegression/Ridge/Lasso/RandomForestRegressor/SVR）。
2. 在上一轮所用的数据集之外，新增 1~2 个结构差异明显的数据集，用 "datasets" 列表给出，
   每项形如 {"data":{"source":"sklearn","name":"digits"},"label":"手写数字"}；
   必须同时保留上一轮原有的数据集（即第一轮单数据集的，也写成 datasets 形式）。
   【数据集必须与该 spec 的 task_type 匹配，否则会被引擎跳过】：
   - classification（分类）：只能选 breast_cancer / digits / wine / iris
   - regression（回归）：只能选 diabetes
   绝不可把回归数据集塞进分类任务（或反之）。
3. 依据上一轮发现（谁赢、谁输、子假设是否成立）重写 3 条更聚焦的子假设。
4. task_type / metric / objective 必须与上一轮保持一致。

★★ 子假设硬约束（与首轮规划器一致，违反会被确定性引擎判为「证据不足」）★★
1. 子假设里提到的「方法」只能是本轮 candidate_methods 中声明过的中文名（含上一轮沿用与本轮新增的），
   严禁出现未声明算法（如高斯朴素贝叶斯、XGBoost、LightGBM 等，除非你已列入 candidate_methods）。
2. 子假设只能围绕本 spec 的 metric 字段（与上一轮相同的 accuracy / f1_macro / r2 / mse）展开，
   禁止改用其它指标（例如上一轮测的是 accuracy，就别去谈「F1 宏平均方差」「AUC」）。
3. 子假设必须是「可量化比较」的句式，引擎只认三类：
   (A) 稳定性类：「在多种子下，<方法甲> 的主指标标准差小于 <方法乙>」；
   (B) 两两比较类：「在主指标上，<方法甲> 显著优于 <方法乙>」；
   (C) 多数据集类（本轮用了多个数据集时）：「<方法甲> 相对 <方法乙> 的优势在数据集 X 上比数据集 Y 上更大」。
4. 禁止写「主假设已验证」「本文证明」等引擎无法核实的表述。

只输出如下 JSON，不要任何解释：
{
  "title": "论文标题",
  "research_question": "更聚焦的机理问题",
  "task_type": "classification 或 regression",
  "metric": "accuracy/f1_macro/r2/mse",
  "objective": "maximize/minimize",
  "datasets": [{"data":{"source":"sklearn","name":"..."},"label":"中文名"}, ...],
  "candidate_methods": [
    {"name":"中文名","estimator":"完整import路径","params":{...}},
    ...
  ],
  "sub_hypotheses": [
    {"id":"H1","statement":"假设","rationale":"为什么","expected":"预期"},
    {"id":"H2",...},{"id":"H3",...}
  ],
  "analysis_plan": "一句话",
  "literature_query": "英文关键词"
}"""


def _result_summary(prev_spec: dict, prev_result: dict) -> str:
    """把上一轮分析结果压成 LLM 可读的简短摘要。"""
    if not isinstance(prev_result, dict):
        return str(prev_result)[:1500]
    lines = []
    if prev_result.get("task_type") == "multi_dataset":
        lines.append("【多数据集结果】")
        for lab, v in (prev_result.get("per_dataset_results") or {}).items():
            lines.append(f"  {lab}: 最优={v.get('best_method')}（{v.get('best_value')}）")
        for m, s in (prev_result.get("rank_stability") or {}).items():
            lines.append(f"  排名稳定性 {m}: 平均排名={s.get('mean_rank')} 标准差={s.get('std_rank')}")
    else:
        lines.append(f"【单数据集结果】最优方法={prev_result.get('best_method')} 主指标={prev_result.get('best_value')}")
        for m in (prev_result.get("metrics") or [])[:12]:
            lines.append(f"  {m.get('name')}: {m.get('value')}")
        for k, v in (prev_result.get("sub_hypothesis_evidence") or {}).items():
            lines.append(f"  {k}: {str(v.get('conclusion', ''))[:140]}")
    return "\n".join(lines)[:1800]


def iterate_spec(prev_spec: dict, prev_result: dict, client=None, model=None, extra=None, round_idx: int = 2):
    """根据上一轮结果产出扩宽/深化的新 spec；无 client 时返回 None。"""
    if client is None:
        return None
    summary = _result_summary(prev_spec, prev_result)
    prompt = (
        _ITERATE_SYSTEM
        + f"\n\n上一轮规格（摘要）：\n{json.dumps(prev_spec, ensure_ascii=False)[:1200]}\n\n"
        + f"上一轮实验结果：\n{summary}\n\n请输出扩宽后的研究规格 JSON（第 {round_idx} 轮）。"
    )
    try:
        spec = _llm_json(client, model, extra, prompt, temperature=0.3)
    except Exception:
        return None
    if not isinstance(spec, dict):
        return None
    return validate_and_fix(spec, prev_spec.get("title", ""), None)


def validate_and_fix(spec: dict, topic: str, data_override: dict | None = None) -> dict:
    """补齐缺省、修正不合法字段，确保引擎能跑。"""
    spec = dict(spec)
    spec.setdefault("title", topic)
    spec.setdefault("research_question", topic)

    # 1) 数据集（兼容单数据集 data 与多数据集 datasets 列表）
    if data_override:
        spec["data"] = data_override
    else:
        ds_list = spec.get("datasets") or spec.get("data_list")
        d = spec.get("data") or {}
        if isinstance(d, dict) and d.get("name"):
            name = d.get("name")
        elif ds_list:
            name = (ds_list[0].get("data") or ds_list[0]).get("name")
        else:
            name = None
        if d.get("source") in ("csv", "parquet", "excel", "feather", "openml"):
            pass  # 用户自带数据，不强制白名单
        else:
            if name in OFFLINE_CLASSIFICATION:
                d = {"source": "sklearn", "name": name}
                spec["task_type"] = "classification"
            elif name in OFFLINE_REGRESSION:
                d = {"source": "sklearn", "name": name}
                spec["task_type"] = "regression"
            else:
                # 兜底：按任务类型选一个
                tt = spec.get("task_type", "classification")
                fallback = OFFLINE_CLASSIFICATION[0] if tt != "regression" else OFFLINE_REGRESSION[0]
                d = {"source": "sklearn", "name": fallback}
        spec["data"] = d
        if ds_list:
            norm = []
            for item in ds_list:
                if isinstance(item, str):
                    nm = item
                    dd = {"source": "sklearn", "name": nm}
                    norm.append({"data": dd, "label": nm})
                elif isinstance(item, dict):
                    dd = item.get("data") or item
                    nm = dd.get("name")
                    if nm in OFFLINE_CLASSIFICATION:
                        dd = {"source": "sklearn", "name": nm}
                    elif nm in OFFLINE_REGRESSION:
                        dd = {"source": "sklearn", "name": nm}
                    norm.append({"data": dd, "label": item.get("label") or nm})
            # 剔除与任务类型冲突的数据集（如把回归数据集塞进分类任务）。
            # 否则该数据集会在引擎里失败并被丢弃，导致标签数与实际参与比较的
            # 数据集数不一致（论文跨数据集表里会出现一整行 "—" 的空值）。
            _tt = spec.get("task_type")
            if _tt in ("classification", "regression"):
                _keep = []
                for it in norm:
                    _nm = (it.get("data") or {}).get("name")
                    if _tt == "classification" and _nm in OFFLINE_REGRESSION:
                        continue
                    if _tt == "regression" and _nm in OFFLINE_CLASSIFICATION:
                        continue
                    _keep.append(it)
                if _keep:
                    norm = _keep
            # 有效数据集少于 2 个即不构成「跨数据集」比较：退回单数据集模式，
            # 避免下游把单个数据集当成多数据集，渲染出只有一行的对比表。
            if len(norm) >= 2:
                spec["datasets"] = norm
            else:
                spec.pop("datasets", None)

    # 2) 任务类型与指标一致性
    tt = spec.get("task_type")
    if tt not in ("classification", "regression"):
        tt = "classification" if spec["data"].get("name") in OFFLINE_CLASSIFICATION else "regression"
        spec["task_type"] = tt
    metric = spec.get("metric")
    if tt == "classification":
        if metric not in ("accuracy", "f1_macro"):
            metric = "accuracy"
        spec["objective"] = "maximize"
    else:
        if metric not in ("r2", "mse", "mae"):
            metric = "r2"
        spec["objective"] = "maximize" if metric == "r2" else "minimize"
    spec["metric"] = metric

    # 3) 候选方法：统一成 dict 写法，补全 estimator
    methods = spec.get("candidate_methods") or []
    fixed = []
    for m in methods:
        if isinstance(m, str):
            full = METHOD_ALIAS.get(m.lower())
            if full:
                fixed.append({"name": m, "estimator": full, "params": {}})
        elif isinstance(m, dict):
            est = m.get("estimator") or METHOD_ALIAS.get((m.get("name") or "").lower())
            if est and est.startswith("sklearn."):
                nm = m.get("name") or est.split(".")[-1]
                params = dict(m.get("params", {}))
                # 需要随机种子但没写占位符的，自动补
                if "<seed>" not in str(params) and any(
                    k in est.lower() for k in ("randomforest", "svc", "gradientboosting",
                                               "ridge", "lasso", "logistic", "forest",
                                               "xgb", "lgbm")
                ):
                    params.setdefault("random_state", "<seed>")
                fixed.append({"name": nm, "estimator": est, "params": params})
    # 3.5) 按 estimator 去重：LLM 可能用不同中文名指代同一方法
    # （例如「支持向量机」与「RBF核SVM」都指向 sklearn.svm.SVC），
    # 若不去重会导致重复方法、跨数据集对比出现空列与重复行。保留首条。
    _seen_est: dict[str, dict] = {}
    _dedup: list[dict] = []
    for m in fixed:
        est = (m.get("estimator") or "").strip()
        if est and est in _seen_est:
            continue
        if est:
            _seen_est[est] = m
        _dedup.append(m)
    fixed = _dedup

    if not fixed:
        fixed = _default_methods(tt)
    spec["candidate_methods"] = fixed

    # 4) 子假设：保证 3 条、含必要字段
    hyps = spec.get("sub_hypotheses") or []
    clean = []
    for i, h in enumerate(hyps, 1):
        if not isinstance(h, dict):
            continue
        clean.append({
            "id": h.get("id") or f"H{i}",
            "statement": h.get("statement") or f"关于第 {i} 个因素的假设",
            "rationale": h.get("rationale") or "",
            "expected": h.get("expected") or "",
        })
    while len(clean) < 3:
        i = len(clean) + 1
        clean.append({
            "id": f"H{i}",
            "statement": f"方法选择对 {metric} 有显著影响（假设 {i}）",
            "rationale": "不同归纳偏置在有限样本下表现不同",
            "expected": "集成/线性方法各有优势，取决于数据尺度与噪声",
        })
    spec["sub_hypotheses"] = clean[:4]

    spec.setdefault("analysis_plan", f"在 {spec['data'].get('name')} 上统一评估各方法，"
                                     f"以 {metric} 为主指标比较，并做多种子稳定性与假设检验。")
    spec.setdefault("literature_query", topic)
    return spec


def _default_methods(task_type: str) -> list[dict]:
    if task_type == "regression":
        return [
            {"name": "线性回归", "estimator": "sklearn.linear_model.LinearRegression", "params": {}},
            {"name": "岭回归", "estimator": "sklearn.linear_model.Ridge", "params": {"random_state": "<seed>"}},
            {"name": "随机森林", "estimator": "sklearn.ensemble.RandomForestRegressor",
             "params": {"n_estimators": 300, "random_state": "<seed>"}},
        ]
    return [
        {"name": "逻辑回归", "estimator": "sklearn.linear_model.LogisticRegression",
         "params": {"max_iter": 3000, "random_state": "<seed>"}},
        {"name": "随机森林", "estimator": "sklearn.ensemble.RandomForestClassifier",
         "params": {"n_estimators": 300, "random_state": "<seed>"}},
        {"name": "支持向量机", "estimator": "sklearn.svm.SVC", "params": {"random_state": "<seed>"}},
    ]


def template_spec(topic: str, data_override: dict | None = None) -> dict:
    """无 API 时的确定性模板 spec（分类 + wine + 3 方法）。"""
    d = data_override or {"source": "sklearn", "name": "wine"}
    tt = "regression" if d.get("name") in OFFLINE_REGRESSION else "classification"
    return {
        "title": topic or "基于多种机器学习方法的基准比较研究",
        "research_question": topic or "不同归纳偏置的方法在给定数据上的表现差异及其成因",
        "task_type": tt,
        "data": d,
        "candidate_methods": _default_methods(tt),
        "metric": "accuracy" if tt == "classification" else "r2",
        "objective": "maximize",
        "sub_hypotheses": [
            {"id": "H1", "statement": "线性方法与非线性方法在主指标上存在显著差异",
             "rationale": "不同归纳偏置对特征尺度与交互的刻画能力不同",
             "expected": "集成方法在复杂边界上更优，线性方法在线性可分时更稳"},
            {"id": "H2", "statement": "特征存在冗余，Top-k 特征可达到接近全量的性能",
             "rationale": "真实数据常含相关特征", "expected": "Top-10 特征性能不显著下降"},
            {"id": "H3", "statement": "模型表现对随机种子具有稳定性",
             "rationale": "稳定训练协议下方差应可控",
             "expected": "多种子标准差小于主指标绝对值的 2%"},
        ],
        "analysis_plan": "统一评估各方法，以主指标比较，并做多种子稳定性与假设检验。",
        "literature_query": topic or "machine learning benchmark comparison",
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：python planner.py \"研究主题\"")
        return 1
    topic = sys.argv[1]
    client, model, extra = llm_client()
    spec = plan_spec(topic, client, model, extra)
    print(json.dumps(spec, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
