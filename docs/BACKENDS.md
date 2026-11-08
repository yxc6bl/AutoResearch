# 补一个领域后端，到底要改哪几处

先纠正一个直觉：这个系统里没有「一个后端文件」可以替换。所谓后端，是散在四个地方的
**接线点**。补一个新领域 = 实现「数据怎么变成 X/y」+「怎么 fit/eval」，让产物符合
`ANALYSIS_JSON` 契约，再在题目解析里加一条目录。论文写作与评审侧原则上**不用改**，
因为它们是按字段契约消费产物的，不认具体领域。

任何时候先跑 `python auto_research.py backends`：它会列出接线点、已装依赖、
四类缺口的优先级和依赖就绪情况。

依赖是否可用，分两级看：`dep_status` 用 `find_spec` 查「文件在不在」，
`--probe` 用子进程**真导入**一遍并给出冷启动耗时（`python auto_research.py backends --probe`）。
后者才作数——torch 这类包会装上却因 DLL / ABI / Python 版本不匹配而 import 失败。
实测：numpy 0.32s、pandas 0.94s、sklearn 2.56s、torch 6.93s、torchvision 13.04s、transformers 3.97s。

## 一、四个接线点

| # | 文件 / 函数 | 要做什么 | 不改会怎样 |
|---|---|---|---|
| ① 题目解析 | `engine/topic_parser.py`：`DATASET_CATALOG`、`_data_binding()`、`detect_task_type()` | 加一条数据目录（领域词、任务类型、模态、来源），让题目能绑到数据 | 题目匹配不上，`benchmark_titles.py` 判不可执行 |
| ② 数据加载 | `engine/analysis_engine.py`：`_load_dataset_inner()`（约 494 行的 `load_dataset` 内）按 `spec.data.source` 分派 | 实现 `_load_xxx(d) -> (X, y, feat_names, task_type)` | 报「不支持的数据源」 |
| ③ 评估协议 | `engine/analysis_engine.py`：`run_analysis()` 分支 + `build_estimator()` 方法工厂 | 决定怎么划分、跑哪些方法、算什么指标、出什么图 | 套错协议（例如把图节点当独立样本随机切）或「没有可运行的候选方法」 |
| ④ 事实契约 | `eval_agent._ground_truth()`、`generate_paper._fact_pack()` | 产物必须给齐字段，见下节 | 正文写「素材未提供」，评审官把缺失当编造 |

已有的四条接线：`tabular`（sklearn/csv/url/openml）、`text_tfidf`（`_run_text`，TF-IDF 折内拟合）、
`timeseries`（滑窗转表格后复用回归主流程）、`clustering`（全量拟合 + ARI/NMI/silhouette）、
`association`（Apriori 关联规则，`_run_association`，纯 Python 无新依赖）。
`timeseries` 是最省事的一种补法：**把新模态转成表格，就完全不用加评估分支**；
`association` 则是另一条路线：**新模态自带评估语义（支持度/置信度/提升度），需独立分支但同样零依赖**。

## 二、ANALYSIS_JSON 字段契约

必需（缺了正文/评审会出问题）：

```
task_type, best_method, best_value, metrics, plots
```

`metrics` 每项必须是 `{"name": str, "value": 数值, ...}`；只给字符串或空列表，
`_fact_pack` 就写不出真实数字。

建议补：`significance`、`seed_variance`、`preprocessing`、`confusion_matrix`、
`per_class`、`findings`、`adaptations`。这些是评审官判「实证严谨度」的证据。

自检：`python auto_research.py backends --check out_x/03_experiment.json`。

## 三、四类缺口，按性价比排

### P1 领域数据缺失 —— 不需要新后端，只补数据

涉及「网络游戏流失玩家预测」「复合材料等效弹性性能」。评估协议跟现有表格
分类/回归完全一样，所以**只动接线点 ①**：找公开 CSV → 放 `data/` →
`DATASET_CATALOG` 加一条（`source: csv`、`path`、`target`、`task`、`domain_kw`）
→ 跑 `benchmark_titles.py` 验收。零模型改动、零新依赖。

坑：本地 `data/_concrete.data` 是老式 `.xls`，`pandas.read_excel` 需要 `xlrd`
或 `python-calamine`。**2026-11-05 已装 `xlrd 2.0.2` 并转成 `data/concrete_strength.csv`**
（1030×9，列名改成 snake_case），目录里加了 `concrete`（`task: regression`、`proxy: true`）：
题目「基于机器学习的复合材料等效弹性性能研究」从不可执行变成**可执行（代理）**，
随机森林 R²=0.8988（7.8s）。之所以标 `proxy`，是因为题目要等效弹性模量、数据给的是抗压强度，
不能冒充同领域数据。

代理标记不是装饰：`_data_binding` 把 `proxy / proxy_note` 写进 spec，
`generate_paper._fact_pack` 与第 3.1 节据此要求正文明确写出「这是代理数据、
数据实际承载的是抗压强度而非弹性模量」。没有这条约束，模型会把抗压强度
直接写成弹性模量——数字全对，结论全错，而且比数字写错更难被发现。
「网络游戏流失」需要另找公开数据（Kaggle 类多需登录，优先考虑 UCI / OpenML）。

### P2 图/网络结构 —— 需要新分支，但不用深度学习栈

涉及「社交网络推荐」「图神经网络过平滑」。关键不是装 GNN 库，而是**评估协议**：
图数据的样本不独立，不能套现有的 `train_test_split` 随机划分，必须按节点切分，
并处理训练时可见整张图的传导泄漏。

最小可跑版本（纯 numpy，无新依赖）：把拓扑压成「一跳邻居均值」特征，拼回节点特征表，
然后复用主流程的分类器。骨架：

```python
def _load_graph(d: dict):
    """source=graph。nodes.csv: node_id, feat..., label；edges.csv: src,dst"""
    import pandas as pd
    nodes = pd.read_csv(d["nodes"]); edges = pd.read_csv(d["edges"])
    id_col, label_col = d.get("node_id", "node_id"), d.get("target", "label")
    y = nodes[label_col].to_numpy()
    feats = [c for c in nodes.columns if c not in (id_col, label_col)]
    X = nodes[feats].to_numpy(dtype=float)
    idx = {n: i for i, n in enumerate(nodes[id_col])}
    A = np.zeros((len(nodes), len(nodes)), dtype=float)
    for s, t in edges[["src", "dst"]].itertuples(index=False):
        A[idx[s], idx[t]] = A[idx[t], idx[s]] = 1.0
    deg = A.sum(1, keepdims=True); deg[deg == 0] = 1
    Xg = A @ X / deg                       # 一跳邻居均值
    return (np.hstack([X, Xg]), y,
            feats + [f"neighbor_{f}" for f in feats], "classification")
```

接线：② 在 `_load_dataset_inner` 加 `if src == "graph": return _load_graph(d)`；
③ 在 `run_analysis` 里按 `source == "graph"` 走节点级 `StratifiedKFold`；
① 目录加一条 `source: graph`、`modality: graph`，`domain_kw` 放「社交网络/图结构/过平滑/节点分类」。

诚实边界：这是**图特征 + 表格分类器**，能接住「结构信息有用吗」这类实证题，
但不是真 GNN。要复现「过平滑」这种现象级结论，得 torch + PyG/DGL + 层数消融，
工作量接近下面的 P3，别把它写成「已支持图神经网络」。

### P3 视觉/视频 —— 最重，先别动

涉及 6 道题。**2026-11-05 已装 `torch 2.15.0+cpu` / `torchvision 0.29.2+cpu`**
（`backends --probe` 实测冷启动 6.93s / 13.04s），依赖不再是障碍；
但还缺预训练权重、图像增广与 CPU 上的训练预算。

技术上最麻烦的不是装库，而是**接口冲突**：现有 `load_dataset` 统一把 X 包成
`DataFrame` 喂给 `build_tabular_preprocessor()`，图像张量走不了这条路。
所以必须加一个独立图像分支，跳过表格预处理，自己管 `DataLoader` / 归一化 / 增广，
最后把结果折算回 `ANALYSIS_JSON`（`metrics` 用 top-1 或 mAP，`preprocessing` 里
如实写权重来源与 epoch）。

注意：`torch` 装完立刻有用的是**表格上的 `PyTorchMLP`**（引擎里早有这个包装器，
只差一个可导入的 torch；实测混凝土数据 R²=0.7011、9.0s），
但这**不是**图像后端——两者不要混为一谈。

建议顺序：先做「小规模图像分类」（Olivetti 那种像素扁平化已能跑，可作为对照组），
再谈检测/分割/视频。不要一上来接「无人收银系统」——那是系统题，不是模型对比题。

### P4 文本 / 语言模型 —— 一半不该塞进这个引擎

`text_tfidf` 已经能跑文本分类。**2026-11-05 已装 `transformers 5.22.1`**
（冷启动 3.97s），但要升级到句向量还需下载权重（几百 MB 到数 GB），
且目前没有任何代码路径调用 transformers——「能 import」不等于「接了语义分支」。

更重要的是，「基于知识图谱的农业知识问答系统」「基于检索增强的大语言模型算法研究」
是**研究方法 / 系统构建题**，产出应是检索链路、消融与人工评测，不是统一基准跑分。
硬塞进确定性 ML 流水线只会得到一份跑题的稿子。这两道建议保留在缺口里，
等 EvidencePack（用户自传证据 + 系统整合计算）那条线做完再说。

## 四、验收清单（每个新后端都要过）

1. `python -m py_compile engine/analysis_engine.py engine/topic_parser.py`
2. `python auto_research.py backends` —— 依赖就绪、缺口优先级符合预期
3. `python auto_research.py backends --check out_x/03_experiment.json` —— 契约通过
4. `python benchmark_titles.py` —— 目标题从不可执行变可执行，且**误匹配 0**；
   若是替代基准数据，必须在目录里标 `proxy`，不许冒充同领域数据
5. `python auto_research.py gates --out out_x` —— G01「分析层产出有效」通过
6. 同题跑两次，`best_method` / `best_value` 一致（随机种子固定，可复现）
7. 若新加评估协议，必须在 `docs/GATES.md` 写明划分方式与泄漏风险，
   否则答辩时「这个数怎么来的」答不上来

## 六、已落地实例：关联规则挖掘（association）

这是照「四接线点」补出来的第一个非 分类/回归/聚类 后端，零新依赖（纯 Python + 已有 sklearn/pandas/matplotlib），
可直接作为后续扩展的模板。改动清单：

- **① 题目解析** `engine/topic_parser.py`
  - `detect_task_type()`：加关联规则关键词（关联规则 / 购物篮 / apriori / 频繁项集 / 共现 / market basket…）→ 返回 `"association"`。
  - `DATASET_CATALOG`：加一条 `source: "association"`、`path` 指向交易 CSV（本仓库 `data/market_basket.csv`，约 4000 笔、25 商品，植入啤酒+尿布等共现结构）。
  - `_data_binding()`：新增 `source=="association"` 分支，返回 `path / min_support / min_confidence`。
  - `_DEFAULT_METHODS["association"] = ["apriori"]`，`build_candidate_methods()` 加 association 分支，`metric_map` 加 `"association": "lift"`，研究问题/假设文本加 association 专用版本。
- **② 数据加载 + ③ 评估协议** `engine/analysis_engine.py`
  - `run_analysis()` 顶部对 `task_type=="association"` 早返回，走独立分支 `_run_association()`。
  - `_load_transactions()`（单列 `items` 逗号分隔 / 多列两种格式）、`_apriori()`（逐层频繁项集 + 先验剪枝）、`_gen_rules()`（默认 1→1 规则，过滤低置信度，算支持度/置信度/lift）。
  - 出图 `assoc_rules.png`（Top 规则按 lift 柱状图），产物严格对齐 `ANALYSIS_JSON` 契约（task_type/best_method/best_value/metrics/plots）。
- **④ 事实契约** `paper/generate_paper.py`
  - `main()` 对 association 直接用 `analysis.best_method/best_value`；`_metric_formula()` 加 lift 分支；`_methodology_math_block()` 加 association 专用数学层（支持度/置信度/lift 定义 + Apriori 算法说明）。评审官 `eval_agent` 只取通用字段，无需改。
- **网站可用性** `web/app.py` + `auto_research.py cmd_plan`
  - `cmd_plan` 增加确定性快路径：题目匹配到本地数据集目录（含 association）时，直接用 `assemble_title_spec` 产出可复现 spec，不依赖 LLM Planner 猜测任务类型。
  - 网站下拉框加入 `market_basket`（映射到 `--data` 本地 CSV），用户也可以直接输入「基于关联规则算法的超市购物篮分析研究」由关键词自动识别。

实测：最优规则为 `啤酒→尿布`（lift=3.8917，支持度 0.2047，置信度 0.8893），共挖出 266 条可解释规则；`backends --check` 契约校验通过。

### 6.1 论文模板必须跟着协议改（否则 G04 硬门会拦）

引擎跑通只解决了「算得出来」。论文能不能交付还差一步：`generate_paper` 的散文模板
默认按监督学习叙事写（多种子方差、配对显著性检验、训练/测试划分），套到确定性算法上
会被 LLM 评审官判成编造。association 这一轮从 `hard_fail` 修到 `hard_pass`，改的就是这些点：

| 位置 | 改法 |
|---|---|
| `_templated_summary()` | 结论句改「提升度最高的规则为 X（lift=…）」，不写方差与显著性 |
| 第 3.3 节 评价协议 | 明示确定性算法：不设随机种子、不做配对检验、统一阈值完整枚举 |
| 5.4 节 | 由「方差/显著性表」改为「规则强度与可解释性」，输出 Top12 规则三指标表（4 位小数） |
| 第 4 节 | 去掉「标准化后划分训练集/测试集」，改为交易级数据说明 |
| 第 6 章标题 | `_deep_cn()`：association →「深入分析」，全文禁用「假设检验」 |
| 8.0 / 8.1 / 8.2 | 创新点用「采用」而非「提出」；局限写 1→1 规则、阈值敏感性、相关非因果 |
| `_fact_pack()` | 注入「不做显著性检验」+ Top12 规则四位小数 + 禁止由指标反推数字 |
| `eval_agent._ground_truth()` | 补 Top12 规则的支持度/置信度/lift，评审官才有据可核 |
| `analysis_engine` metrics.setting | 与事实清单同精度（4 位小数），避免正文 0.205 与事实 0.2047 打架 |

一句话：**新协议要同时改「事实清单」和「叙事模板」**。只改引擎不改模板，
产物会被 G04 判成编造；这不是模型能力问题，是模板在说谎。

### 6.2 写作参与证据落盘（G03）

`generate_paper` 结束时同时写 `06_paper/WRITE_STATUS.json`（`llm_written` / `llm_sections` /
`model` / 时间戳），并在 stdout 打一行 `@@WRITE_STATUS@@ {json}` 哨兵。
`writing_agent` 优先读该文件，`mark_delivery` 和 G03 门也以它为准。

原因：旧实现靠 `stdout.rfind("{")` 反解，stdout 里一旦混入网关告警或进度行就会解析失败，
`llm_written` 变成 `null`，G03 只能判「未验证」，最终 `07_gate_report.json` 的
`deliverable` 会与 `DELIVERY_STATUS.json` 互相矛盾。

## 五、一句话结论

补后端不是换模型，是**把新模态接到四个接线点上，并让产物说真话**。
最省的是「转成表格复用主流程」（时序已经这么做了），最贵的是「张量 / 图 / 生成式」，
因为它们连数据划分和指标定义都要换。先做 P1 和 P2，别先碰 P3/P4。

## 七、已落地实例：图节点分类（graph）

第二个「菜单外」后端，用来验证四接线点不是只为 association 定制的。

### 7.1 数据：Cora 真实引文网络

`data/cora.content`（节点表）+ `data/cora.cites`（边表），来自 LINQS / Planetoid 的公开版本
（tkipf/pygcn 仓库的 data/cora，下载时校验 Content-Length：7,823,427 字节）。
实测：**2708 个节点、1433 维词袋特征、7 类论文主题、5429 条引用边，端点零缺失**。
第一次下载只拿到 929 行（curl 超时截断），是校验行数时才发现的——数据完整性也要验收。

### 7.2 四个接线点

| 接线点 | 落点 |
|---|---|
| ① 题目解析 | `DATASET_CATALOG` 加 `cora`（`source=graph`、`format=linqs`、`modality=graph`、`methods=[lr,nb,dt,knn]`、`n_splits=5`）；`_data_binding()` 加 graph 分支；`build_candidate_methods()` 加方法白名单（1433 维稀疏特征不跑 RBF-SVC） |
| ② 数据加载 | `_load_graph()`：支持 `linqs`（id+特征+label / src,dst）与 `csv`（nodes.csv+edges.csv）两种格式，构造对称去自环邻接 A 与度归一化 Â |
| ③ 评估协议 | `run_analysis()` 按 `data.source=="graph"` 早返回 `_run_graph()`：节点级 `StratifiedKFold(5, shuffle, seed=42)` × 三种特征协议 × 4 个分类器 |
| ④ 事实契约 | `_fact_pack()` / `eval_agent._ground_truth()` 注入 12 个组合的四位小数、配对 t 检验 p 值与传导泄漏说明；模板在 3.3 / 3.3.1 数学层 / 第 4 章 / 5.4 / 8.0 / 8.1 / 8.2 全部按 `analysis.graph` 分支 |

### 7.3 三种特征协议：把泄漏量化出来

```
X^(self) = X
X^(full) = [X, ÂX]              划分前用全图特征 → 传导泄漏
X^(fold) = [X, Â·mask_tr(X)]    逐折只用训练折特征 → 无泄漏
```

**核心结论只从无泄漏协议里选。** 系统宁可报 0.8515，也不把含泄漏的 0.8641 写成最优。
这是 D7（模板随协议分支）之外的新一层：**结论的选取规则本身也要写进引擎**。

实测（5 折，4 方法，67 秒，12 项指标，1 张图）：

| 项目 | 数值 |
|---|---|
| 最优无泄漏组合 | 逻辑回归｜折内邻居均值，accuracy=0.8515、macro-F1=0.8381 |
| 含泄漏上界 | 逻辑回归｜全图邻居均值 0.8641（**排除**） |
| 泄漏虚高幅度 | 全图相对折内平均 +0.0117 |
| 拓扑特征增益 | 逻辑回归逐折差值 +0.0838，p=0.00037；四个方法均显著 |
| 泄漏是否显著 | 逻辑回归差值 +0.0125，p=0.0701（不显著）；朴素贝叶斯 +0.0074，p=0.0270（显著） |

### 7.4 模板必须跟着改（第二次验证同一条经验）

默认模板说「5 个随机种子（42–46）」「按固定比例划分训练集与测试集」
「McNemar 配对检验」「独立同分布样本」「中位数插补 + 标准化」——
在图任务上**每一句都是假的**。改动清单与 6.1 同构：

- `_methodology_math_block()` 增 graph 分支：式(2)–(4) 三种特征协议、式(5) 只在无泄漏协议里取最优；
  明确写「本文用逐折配对 t 检验，不是 McNemar」。
- 3.3 评价协议 / 第 4 章 / `round_txt`：改节点级分层 K 折，消除「第 None 轮」。
- 5.1 表头改「方法×特征协议 / 评估口径」；5.4 用 `_graph_variance_table()`，表头是「各折取值」而非「各种子取值」。
- 8.0 / 8.1 / 8.2：创新点写「把泄漏与增益分开度量」，局限写明非真 GNN、传导式设定、折数有限。
- `_fact_pack()` 与 `_ground_truth()`：注入真实数字 + 三条硬约束（禁用随机种子/标准化/GNN 表述，
  核心结论只准用指定组合，全图协议只能作上界对照）。

### 7.5 网站入口

`web/app.py` 的下拉框加 `cora_graph`，`run_pipeline` 把它翻译成 `--catalog-key cora`
（图数据要 nodes+edges 两个文件，不能走 `--data/--dataset` 的单表入口）；
`auto_research.py plan` 新增 `--catalog-key`，从 `DATASET_CATALOG` 按 key 取真实绑定，
并对非表格源强制走确定性 spec 路径，避免 LLM Planner 把它当 sklearn 数据集。

诚实边界：Cora 是**真实**引文网络，但对「社交网络推荐」「GNN 过平滑」这两道题库真题仍是
**代理数据**（目录里标 `proxy`）；而且本后端是一跳邻居特征 + 表格分类器，不是图神经网络。
题目明确研究引文网络节点分类时，`assemble_title_spec` 会按题目语义去掉代理标记，
否则正文会对着真实数据写「这是代理数据」，反过来变成一句假话。

### 7.6 G04 硬门：从 10 条幻觉到 hard_pass

第一版论文 `hard_fail`、94.8 分但 10 条幻觉。逐条核对后发现**9 条都是假阳性**：
正文里的逐类指标、混淆矩阵、逐折准确率全部来自 `03_experiment.json` 的实测值，
但评审官的 `_ground_truth()` 把 `seed_variance` 截到 200 字（只露出第一个方法的折值），
而且压根没把 `per_class` / `confusion_matrix` 交给评审官——**事实清单比正文短，
真数也就成了「无据」**。第 10 条是真错：正文用 0.8641−0.8515 自己算出「虚高 0.0126」，
而配对差值由未四舍五入的逐折准确率算出，实测是 0.0125。

修法（都在系统层，模型没换）：

- `eval_agent._ground_truth()`：`seed_variance` 上限 200→2000 字；
  逐类指标、混淆矩阵全量进事实清单；图实验显式列出 12 个组合的逐折准确率。
- `_fact_pack()` 加硬约束：协议差值必须原样引用 `paired_tests.diff`，
  严禁自行用四位小数均值相减推算；逐类/逐折/混淆矩阵数字只能引用清单给定值。
- 已交付稿同步修正两处 `0.0126 → 0.0125`，并补上图注缺失的图片引用
  （`_plots_block` 现在会回退查 `<out_dir>/plots/<file>`，以前只查根目录，正文只剩图注没有图）。

复评结果：**`hard_pass`、综合 94.8、评审官 85 分、幻觉 0 条**；
`gates` 的 P0 四项全通过，结论「可交付」。中间有一次评审官返回 `bad_json`，
G04 正确记为「未验证」而不是通过，重跑一次才拿到有效评审——闸门没放水。

产物：`out_graph_cli/06_paper/paper_full.md`（79KB，15245 汉字）、
`paper_full.docx`（97KB）、`plots/graph_protocol.png`（34KB）。
网站 `/api/open?dir=out_graph_cli` 返回 `hard_pass / 94.8`，正文含
「Cora / 传导泄漏 / 0.8515 / 节点级分层」与图片路径。

## 八、已落地实例：小规模图像分类（image）

### 8.1 真实数据与可复现性

`data/mnist/` 下四个 IDX gzip 文件，来自 torchvision 使用的公开 MNIST 镜像。
下载后逐个校验 magic 与维度：2051 / 60000×28×28、2051 / 10000×28×28、
2049 / 60000、2049 / 10000，端点齐全。

实验沿用 MNIST 官方划分，按固定种子 42 分层抽取 12000/2000 张（控制 CPU 预算、
保持各类比例）。**torch 多线程下卷积的浮点归约顺序不确定**，实测同题两次跑出
0.9665 / 0.9685；固定 `torch.set_num_threads(1)` + `manual_seed(42)` 后两次都是 0.9615。
这是第一个 torch 后端特有的坑：确定性不只要设种子，还要限制并行归约。

### 8.2 四接线点

| 接线点 | 落点 |
|---|---|
| ① 题目解析 | `DATASET_CATALOG` 加 `mnist`（`source=image`、`format=idx`、`modality=image`、方法白名单、`strict_kw`）；`_data_binding` 加 image 分支；`build_candidate_methods` 支持白名单 |
| ② 数据加载 | 独立模块 `engine/image_backend.py` 的 `load_idx()` 直读 gzip IDX 张量 (N,1,28,28)，**跳过** `build_tabular_preprocessor`（它只认 DataFrame） |
| ③ 评估协议 | `run_image()`：三个模型共用同一划分 / 种子 / 训练预算，自管 batch、epoch、参数量与耗时；指标 top-1 accuracy + macro-F1，出三张图 |
| ④ 事实契约 | `_fact_pack` / `_ground_truth` 注入模型实测值、逐 epoch 曲线、参数量与训练配置；模板在 3.3 / 数学层 / 第 4 章 / 5.1 / 5.4 / 第 6 章标题 / 8.0 / 8.1 / 8.2 全部按 `analysis.image` 分支 |

三个模型：小型 CNN（2 层卷积 + 池化 + 全连接，105866 参数）、扁平像素 MLP
（101770 参数）、像素逻辑回归（7850 参数）。5 个 epoch、batch 128、
CPU 从头训练、无预训练权重、增广接口预留但默认关闭、预处理只有 /255。

### 8.3 一个必要的防守：`strict_kw`

刚加 MNIST 目录时用了「卷积 / cnn / 像素 / 图像分类」这类泛关键词，
`match_dataset` 里 `modality=image` 还有 +1 加权，结果把车辆识别、域迁移、
乳腺诊断等题从它们真正的同领域数据集上抢了过来：
**同领域真实可执行从 14 掉到 10**。

修法是给目录加 `strict_kw`：题目必须点名「手写数字 / 数字识别 / mnist / 手写体」
才允许匹配 MNIST。再去掉多余的 `proxy` 标记（strict_kw 保证它接住的都是同领域真实数据），
benchmark 恢复 **21/33、代理 7、同领域真实 14、误匹配 0**。

一句话：**扩能力不能以牺牲已有正确匹配为代价**，每次加目录都要重跑
`benchmark_titles.py` 看「同领域真实」这一栏有没有掉。

### 8.4 网站入口

下拉框加 `mnist_image` → `--catalog-key mnist`；`plan` 对 image 也强制走确定性 spec
（否则 LLM Planner 会把它当 sklearn 数据集）。

### 8.5 全链路验收：图像论文通过 G04

`plan --catalog-key mnist --no-paper` 抓到 30 篇文献（30/30 带 LM 阅读判断）
并写出实验产物 → `step write`（16415 汉字）→ `step eval`。

第一次写作撞上网关瞬时探测失败，生成器走了确定性骨架稿（5422 字、
`llm_written=false`）。重跑前顺手修掉评审官在骨架稿里指出的三处系统性措辞：

- 逻辑回归的求解器名（SAGA）是模型自己编的 → 引擎把真实 `lr.solver='lbfgs'`
  与 `max_iter=200` 写进产物，事实清单与评审清单同步暴露，不再靠猜。
- 8.0 创新点「新增独立图像分支 / 进入统一事实契约」被判夸大 → 改为
  「采用独立的图像分类流程……使指标成为可核对的实验结果」。
- 8.2 展望提「接入迁移学习」被判易误导 → 改为明确写「上述均为后续工作，
  本文未使用预训练权重、迁移学习与检测链路」；同时把事实清单里的禁令
  收紧成「声称【本文】使用」才算编造，避免把正常的未来展望也判成幻觉。

复评：**`hard_pass`、综合 99.2、评审官 78 分、0 幻觉**；
`gates` 的 P0 四项全通过，结论「可交付」。第 6 章标题按图像实验改为「深入分析」，
三张图都真正嵌进正文与 Word（此前 `_plots_block` 只查根目录，只剩图注）。

产物：`out_image_cli/06_paper/paper_full.md`（73KB）、`paper_full.docx`（208KB）、
三张图（75KB / 37KB / 37KB）。网站 `/api/open?dir=out_image_cli` 返回
`hard_pass / 99.2`、40625 字符正文与三张图的相对路径。
