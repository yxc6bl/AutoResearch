# 能力现状盘点（2026-11-08 实测）

> 这份文件只写磁盘上真实存在、命令能复跑的东西。设想、计划、没验证的都放在最后一节。
> 复核命令：`python auto_research.py backends --probe`、`python benchmark_titles.py`、
> `python auto_research.py gates --out <目录>`。

## 一句话

给一句中文题目，本地确定性引擎跑真实实验 → 生成事实清单 → 模型按清单写中文论文 →
11 道闸门加一个 LLM 评审官判定「能不能交」。**系统的价值不在生成，在拒绝交付。**

## 一、七条已接线后端

| 任务形态 | 真实数据 | 评估协议 | 实测结果 |
|---|---|---|---|
| 表格分类 / 回归 | UCI 心脏病、NASA 翼型、混凝土、sklearn 内置 | 5 种子 + 配对检验 + 消融 | 心脏病 0.8033；混凝土 R²=0.8988 |
| 文本分类 | 20 类新闻组、本地 text_sample | TF-IDF 在折内拟合防泄漏 | 已接线，本轮未单独出论文 |
| 时序预测 | 墨尔本 1981–1990 每日最低气温 3650 行 | 滑窗 lag7 + 季节编码转表格，复用回归 | 梯度提升 R²=0.5647，26.9s |
| 聚类 | 鸢尾花 | 全量拟合，ARI/NMI/silhouette | KMeans ARI=0.6201，2.8s |
| 关联规则 | market_basket 交易 CSV | Apriori 逐层枚举，支持度/置信度/lift | 啤酒→尿布 lift=3.8917，266 条规则 |
| 图节点分类 | Cora 真实引文网络 2708 节点 / 5429 边 | 节点级 5 折 × 三种特征协议，显式量化传导泄漏 | 无泄漏最优 0.8515；含泄漏 0.8641 被排除 |
| 图像分类 | MNIST IDX 60000/10000 张 | 固定划分 + 单线程 torch CNN/MLP/线性基线 | CNN 0.9615，5 epoch，20s |

题库 33 道真实高校题：**21 道可执行（63.6%）**，其中 7 道代理数据，
**同领域真实数据 14 道**，误匹配 0。数字没有靠放宽关键词刷上去。

## 二、五篇实测论文（产物都在本机，可复评）

| 目录 | 类型 | 闸门 | 综合分 | 评审官 | 幻觉 | 正文字数 | 结论 |
|---|---|---|---|---|---|---|---|
| `out_assoc` | 关联规则 | hard_pass | 98.3 | 76 | 0 | 16009 | 可交付 |
| `out_web/task_0001` | 网站端到端跑出来的关联规则 | hard_pass | 98.7 | 78 | 0 | 14781 | 可交付 |
| `out_graph_cli` | 图节点分类 | hard_pass | 94.8 | 85 | 0 | 15245 | 可交付 |
| `out_image_cli` | 图像分类 | hard_pass | 99.2 | 78 | 0 | 16415 | 可交付 |
| `out_gate_real` | 反例：乳腺癌分类 | **hard_fail** | 99.5 | 74 | 5 | 17064 | **不可交付** |

最后一行是特意留的证据：规则评分给了 99.5「优秀」，评审官抓出 5 条事实矛盾，
闸门照样拒交。**加权分不能决定能不能交。**

## 三、质量闸门（11 道，P0 四项）

- G01 分析层产出有效；G02 正文汉字 ≥6000；G03 模型确实参与写作（`WRITE_STATUS.json` 落盘）；
  G04 评审官已介入且零幻觉。
- 评审官缺席记 `unknown` 而不是 `pass`；`bad_json` 那次实测被 G04 拦成「未验证」，没放水。
- 评审闭环最多 3 轮，按 `(分数, -幻觉数)` 取最优轮回滚，md / verdict / docx 三者同源。
- 横向台账：`00_trace.jsonl`（阶段耗时）、`memory/feedback.jsonl`（113 条反馈、
  同题通过后可销账）、`07_gate_approvals.json`（人工放行留名留理由）。
- 每加一种新协议，模板与事实清单必须一起改。关联规则那轮从 4 次 `hard_fail` 到
  `hard_pass`、图论文 10 条幻觉里 9 条是「事实清单被截断」的假阳性，都写进了
  `docs/BACKENDS.md` 与 `docs/GATES.md`。

## 四、网站（用户实际操作路径）

填 API Key（必填，只用于本次运行、不保存）→ 填题目 / 选数据集 → 点运行 →
看论文全文、评分卡、文献、指标、图表；「打开已有结果」可以不开 Key 直接看已生成的成果。

下拉 9 项：`breast_cancer`、`iris`、`wine`、`digits`、`load_diabetes`、
`fetch_california_housing`、`market_basket`（关联规则）、`cora_graph`（图）、
`mnist_image`（图像）。图与图像走 `--catalog-key`，不经过单表入口。

实测：`/api/open?dir=out_image_cli` 返回 `hard_pass / 99.2`、40625 字符正文与三张图；
`POST /api/run` 从零跑完一条 25 分钟链路，产出 98.7 分可交付论文。

## 五、依赖与环境

Python 3.13；numpy 2.5.4 / pandas 3.1.0 / scikit-learn 1.9.1 /
matplotlib 3.11.2 / torch 2.15.0+cpu / torchvision 0.29.2+cpu /
transformers 5.22.1 / networkx 3.7 / xlrd 2.0.2。
`python auto_research.py backends --probe` 用子进程真导入并计时，
`find_spec` 只算「文件在」，不算「能用」。

## 六、明确还做不到（别写成已完成）

- 目标检测、语义分割、视频理解：要换数据划分和指标定义，不是加个模型。
- 真图神经网络 / 过平滑复现：需要 torch_geometric、层数消融与同领域图数据。
- RAG、知识问答：本质是检索链路加人工评测，硬塞进确定性跑分流水线会跑题。
- 领域数据缺口：网络游戏流失、复合材料等效弹性模量、社交网络推荐等仍是代理数据。
- 创新性、学术审美、导师判断：系统只保证「每句话有据」，不保证「这句话值得说」。
- 语义文本 / transformers：库装好了，但**没有任何代码路径调用**，不算已支持。

## 七、共享包

`python scripts/make_share_package.py` 会生成 `dist/AutoResearch_share_<日期>.zip`：
完整代码、公开数据集、全部文档、三篇可交付论文示例（含 Word 与图）、
一键启动脚本 `start_web.py`、`.env.example`。排除 `.git/`、`out_*/`、密钥、
运行日志与半成品。解压后 `python start_web.py` 即可打开网站。
