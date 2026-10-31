# 任务 Spec 契约（SPEC_SCHEMA）

`auto_research.py` 的唯一输入是一份 JSON「任务规格」。任何 agent（LLM planner / 人类研究者 / 上游调度器）只要能产出符合本契约的 JSON，就能让本系统**端到端**完成「加载数据 → 跑候选方法 → 跨种子统计 → 出图 → 生成中文论文」，无需改动任何引擎代码。

> 引擎对数据集与方法**零硬编码**：spec 里声明什么就跑什么。

---

## 1. 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 否 | 任务标识（用于产物目录 / 引用） |
| `direction` | string | 否 | 研究方向的一句话描述（进论文标题） |
| `topic` | string | 否 | 主题（无 `direction` 时作标题用） |
| `research_question` | string | 否 | 机制化研究问题（进论文「问题设定」） |
| `hypothesis_statement` | string | 否 | 主假设 |
| `sub_hypotheses` | list[string] | 否 | 子假设列表，长度任意；引擎会自动为每个生成「有凭有据」的裁决 |
| `analysis_plan` | string | 否 | 分析重点（机制化） |
| `metric` | string | 是 | 主指标名：`accuracy` / `f1_macro` / `r2` / `mse` / `mae` 等 |
| `secondary_metrics` | list[string] | 否 | 次要指标 |
| `objective` | string | 否 | `maximize`（默认）或 `minimize` |
| `data` | object | 是* | 数据集来源声明（见 §2） |
| `dataset` | string | 否* | 向后兼容：写 sklearn 数据集名（如 `digits`）或内置 `breast_cancer/digits/diabetes` |
| `candidate_methods` | list | 是 | 候选方法声明（见 §3） |
| `task_type` | string | 否 | `classification` / `regression` / `clustering`；可覆盖数据加载器推断 |

> `data` 与 `dataset` 至少给一个。优先使用 `data.source` 显式声明。

---

## 2. `data`：数据集来源

引擎支持的数据源：

### 2.1 任意 sklearn 数据集（推荐做快速验证）
```json
"data": { "source": "sklearn", "name": "wine" }
```
`name` 为 `load_<name>` 对应的数据集（如 wine / iris / breast_cancer）。`task_type` 可显式指定，否则按目标值推断。

### 2.2 本地表格文件（最贴近真实场景）
```json
"data": {
  "source": "csv",            // csv / parquet / excel / feather
  "path": "data/my_dataset.csv",
  "target": "label",          // 列名，或整数列索引
  "task_type": "classification"   // 可选，缺省自动推断
}
```

### 2.3 OpenML（需联网）
```json
"data": { "source": "openml", "dataset_id": 61, "version": 1 }
```

### 2.4 内置三件套（向后兼容）
不写 `data`，直接 `"dataset": "digits"` 等同 `{ "source": "sklearn", "name": "digits" }` 的特例。

---

## 3. `candidate_methods`：候选方法

支持两种写法，**可混用**。

### 3.1 旧式字符串（内置注册表，向后兼容）
```json
"candidate_methods": ["LogisticRegression", "RandomForestClassifier", "SVC"]
```
支持：分类 `LogisticRegression / RandomForestClassifier / GradientBoostingClassifier / SVC / KNeighborsClassifier`；回归 `LinearRegression / Ridge / Lasso / ElasticNet`。

### 3.2 新式字典（任意 sklearn 估计器，零硬编码）
```json
"candidate_methods": [
  { "name": "rf", "estimator": "sklearn.ensemble.RandomForestClassifier",
    "params": { "n_estimators": 300, "random_state": "<seed>" } },
  { "name": "xgb", "estimator": "xgboost.XGBClassifier",
    "params": { "n_estimators": 300, "random_state": "<seed>" } }
]
```
- `estimator`：点分导入路径（`模块.类`），引擎动态 `import` 并实例化。
- `params`：构造参数；值为 `"<seed>"` 的字符串会被替换为本轮随机种子（5 个种子：42–46）。
- `name`：可选，作为该方法在指标/证据里的键；缺省用 `estimator` 全名。

> 只要 `import` 得到的类有 `fit/predict`，且分类器有 `predict`、回归器有 `predict`，引擎即可驱动；特征重要性自动取 `feature_importances_` 或 `coef_`。

### 3.3 多后端支持（sklearn / xgboost / lightgbm / PyTorch）

引擎通过动态 `import` 支持满足 `fit/predict` 接口的估计器；聚类是独立分支，支持内置的 KMeans、AgglomerativeClustering、DBSCAN 名称及其点分导入路径字典。聚类字典可通过 `params` 覆盖内置参数（如簇数、距离度量、DBSCAN 的 `eps`）：

| 后端 | `estimator` 写法 | 需要安装的包 |
| --- | --- | --- |
| sklearn | `sklearn.ensemble.RandomForestClassifier` | 已内置 |
| XGBoost | `xgboost.XGBClassifier` / `xgboost.XGBRegressor` | `pip install xgboost` |
| LightGBM | `lightgbm.LGBMClassifier` / `lightgbm.LGBMRegressor` | `pip install lightgbm` |
| PyTorch | `PyTorchMLP`（内置 sklearn 兼容包装器） | `pip install torch` |

> 若某常规方法无法导入/实例化，会记入 `skipped_methods`；训练失败也会按方法和种子记录原因。所有候选方法失败时实验以错误结束，不应将空结果视作成功。
>
> PyTorch MLP 示例（二分类/多分类自动识别）：
> ```json
> { "name": "mlp", "estimator": "PyTorchMLP",
>   "params": { "hidden": [64, 32], "epochs": 60, "lr": 0.001, "task": "classification" } }
> ```

---

## 4. 最小可运行示例

```json
{
  "direction": "葡萄酒品种判别",
  "research_question": "三种葡萄酒的化学成分在线性与集成模型下的可分性差异？",
  "data": { "source": "sklearn", "name": "wine" },
  "metric": "accuracy",
  "objective": "maximize",
  "candidate_methods": ["LogisticRegression", "RandomForestClassifier", "SVC"],
  "sub_hypotheses": [
    "H1（稳定性）：随机森林的多种子标准差不大于逻辑回归。",
    "H2（特征贡献）：Top-10 特征已能逼近全量精度，说明特征冗余。",
    "H3（误差结构）：错误集中在化学成分相近的品种对上。",
    "H4（正则化）：逻辑回归加入正则后测试准确率不显著下降。"
  ]
}
```

运行：
```bash
python auto_research.py run --spec my_spec.json --out ./out_mytask
```

---

## 5. 完整示例（含字典式方法 + 文件数据源）

```json
{
  "id": "house-price",
  "direction": "房价回归的可解释性机理",
  "research_question": "在加州房价数据上，线性模型的系数稀疏性与集成模型的偏差-方差权衡如何？",
  "hypothesis_statement": "房价主要由少数区位/房龄特征线性解释，Lasso 能将大量系数压缩为 0。",
  "sub_hypotheses": [
    "H1（稀疏性）：Lasso 将过半系数精确压缩为 0。",
    "H2（稳定性）：集成模型多种子 R² 标准差不大于线性回归。",
    "H3（正则化）：Ridge/Lasso 相对普通最小二乘的测试 R² 有显著改善。",
    "H4（特征贡献）：少数 Top 特征主导 R²。"
  ],
  "analysis_plan": "报告各方法 5 种子 R²±std、系数表、残差统计、稀疏性、Top-k 特征消融。",
  "metric": "r2",
  "secondary_metrics": ["mse", "mae"],
  "objective": "maximize",
  "data": { "source": "sklearn", "name": "california_housing" },
  "candidate_methods": [
    "LinearRegression", "Ridge", "Lasso", "ElasticNet",
    { "name": "rf", "estimator": "sklearn.ensemble.RandomForestRegressor",
      "params": { "n_estimators": 300, "random_state": "<seed>" } }
  ]
}
```

---

## 6. 引擎保证（与 spec 无关）

- 固定 5 个随机种子（42–46）复现，报告主指标均值±标准差。
- 自动产出：逐类指标 / 混淆矩阵 / 特征重要性 / 多种子方差 / 最优-次优配对显著性检验 / 3 张图。
- 若 `sub_hypotheses` 存在，自动为每条生成基于真实实验数据的裁决证据（过拟合 gap、多种子稳定性、正则化对照、特征选择消融、稀疏性零系数等）。
- 论文生成层（generate_paper）任务无关，仅消费上述产物。

以上五种子/配对检验说明适用于分类和回归；聚类使用全量数据单次拟合，默认以 ARI 排序，支持 `ari`、`nmi`、`silhouette` 指标，不报告配对显著性检验。

> 因此：换一份 spec（不同数据集 / 不同方法 / 不同假设），即可产出一篇新的、有真实数字支撑的论文——这就是「通用 Auto Research」而非「自娱自乐」。
