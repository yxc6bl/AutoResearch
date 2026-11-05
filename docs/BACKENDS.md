# 补一个领域后端，到底要改哪几处

先纠正一个直觉：这个系统里没有「一个后端文件」可以替换。所谓后端，是散在四个地方的
**接线点**。补一个新领域 = 实现「数据怎么变成 X/y」+「怎么 fit/eval」，让产物符合
`ANALYSIS_JSON` 契约，再在题目解析里加一条目录。论文写作与评审侧原则上**不用改**，
因为它们是按字段契约消费产物的，不认具体领域。

任何时候先跑 `python auto_research.py backends`：它会列出接线点、已装依赖、
四类缺口的优先级和依赖就绪情况。

## 一、四个接线点

| # | 文件 / 函数 | 要做什么 | 不改会怎样 |
|---|---|---|---|
| ① 题目解析 | `engine/topic_parser.py`：`DATASET_CATALOG`、`_data_binding()`、`detect_task_type()` | 加一条数据目录（领域词、任务类型、模态、来源），让题目能绑到数据 | 题目匹配不上，`benchmark_titles.py` 判不可执行 |
| ② 数据加载 | `engine/analysis_engine.py`：`_load_dataset_inner()`（约 494 行的 `load_dataset` 内）按 `spec.data.source` 分派 | 实现 `_load_xxx(d) -> (X, y, feat_names, task_type)` | 报「不支持的数据源」 |
| ③ 评估协议 | `engine/analysis_engine.py`：`run_analysis()` 分支 + `build_estimator()` 方法工厂 | 决定怎么划分、跑哪些方法、算什么指标、出什么图 | 套错协议（例如把图节点当独立样本随机切）或「没有可运行的候选方法」 |
| ④ 事实契约 | `eval_agent._ground_truth()`、`generate_paper._fact_pack()` | 产物必须给齐字段，见下节 | 正文写「素材未提供」，评审官把缺失当编造 |

已有的四条接线：`tabular`（sklearn/csv/url/openml）、`text_tfidf`（`_run_text`，TF-IDF 折内拟合）、
`timeseries`（滑窗转表格后复用回归主流程）、`clustering`（全量拟合 + ARI/NMI/silhouette）。
`timeseries` 是最省事的一种补法：**把新模态转成表格，就完全不用加评估分支**。

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
或 `python-calamine`，当前都没装；最省事是找同数据集的 CSV 版本。
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

涉及 6 道题。`torch` / `torchvision` 当前**都没装**，还要预训练权重、图像增广、
CPU 上按小时算的训练时间。

技术上最麻烦的不是装库，而是**接口冲突**：现有 `load_dataset` 统一把 X 包成
`DataFrame` 喂给 `build_tabular_preprocessor()`，图像张量走不了这条路。
所以必须加一个独立图像分支，跳过表格预处理，自己管 `DataLoader` / 归一化 / 增广，
最后把结果折算回 `ANALYSIS_JSON`（`metrics` 用 top-1 或 mAP，`preprocessing` 里
如实写权重来源与 epoch）。

建议顺序：先做「小规模图像分类」（Olivetti 那种像素扁平化已能跑，可作为对照组），
再谈检测/分割/视频。不要一上来接「无人收银系统」——那是系统题，不是模型对比题。

### P4 文本 / 语言模型 —— 一半不该塞进这个引擎

`text_tfidf` 已经能跑文本分类。要升级到句向量需 `transformers` + 权重（几百 MB 到数 GB），
`torch` 也没装。

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

## 五、一句话结论

补后端不是换模型，是**把新模态接到四个接线点上，并让产物说真话**。
最省的是「转成表格复用主流程」（时序已经这么做了），最贵的是「张量 / 图 / 生成式」，
因为它们连数据划分和指标定义都要换。先做 P1 和 P2，别先碰 P3/P4。
