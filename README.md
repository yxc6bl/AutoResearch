# AutoResearch

基于 JSON 任务规格运行确定性机器学习分析，并可生成中文 Markdown/DOCX 论文。实际入口为 `auto_research.py`；代码结构以仓库现存文件为准。

## 结构

```text
auto_research.py             # run / plan / iterate 命令入口
planner.py                   # 主题规划（调用 LLM）
literature.py                # 文献检索
engine/analysis_engine.py    # 数据加载、训练、统计和绘图
engine/tasks/proof/          # 可运行任务规格和样例数据
paper/generate_paper.py      # 消费流水线 JSON 产物并生成论文
requirements.txt             # 运行依赖
SPEC_SCHEMA.md               # spec 字段说明
```

仓库没有 `run.py`、`engine/experiment_agent.py`、`engine/_sandbox_guard.py`、`paper/orchestrator.py` 或文档所称的中文 Agent 目录。实验由 `analysis_engine.py` 本地执行，不是 LLM 编写代码或沙箱执行。

## 安装与运行

建议使用 Python 3.11。安装完整依赖：

```bash
python -m pip install -r requirements.txt
python auto_research.py run --spec engine/tasks/proof/wine_sklearn_spec.json --out out_wine
```

仅运行分析（不生成论文）：

```bash
python auto_research.py run --spec engine/tasks/proof/wine_sklearn_spec.json --out out_wine --no-paper
```

其他命令：`python auto_research.py plan --topic "..."`、`python auto_research.py iterate --topic "..." --rounds 3`。`plan`/`iterate` 的规划和论文润色可能调用 `.env` 中配置的 LLM；直接 `run` 的分析不需要 API Key，LLM 不可用时论文生成器可回退到模板。

## 论文类型（`--paper-type`）

系统支持**六种**可插拔论文类型，`plan` / `run` / `from_title` 都接受 `--paper-type`
（默认 `method_comparison`），网页端在「论文类型」下拉里提供同样的六项：

| `paper_type` | 含义 | 是否跑确定性实验 | 章节骨架 | 零编造约束 |
|---|---|---|---|---|
| `method_comparison` | 方法对比型实证研究 | 是（训练/评估/统计检验） | 8 章实证骨架 | 数字全部来自实验 JSON |
| `survey` | 文献综述型 | 否（只抓文献、做分类梳理） | 绪论→检索方法→分类框架→研究现状→趋势争论→开放问题→结论 | 只引用 `01_literature.json` 真实文献 |
| `system_design` | 系统/软件设计型 | 否（需求→架构→模块→验证） | 同左，7 章 | 只依据 `system_spec` |
| `case_study` | 案例研究型 | 否（理论框架→案例剖析→启示） | 绪论→理论回顾→研究设计→案例背景→案例分析→讨论→结论 | 只依据 `case_spec` + 真实文献 |
| `technical_report` | 工程实践报告型 | 否（需求→方案→实施→验收） | 概述→需求约束→方案设计→实施→结果验证→问题改进→总结 | 只依据 `project_spec` + 真实文献 |
| `research_proposal` | 研究方案设计型（开题） | 否（问题→现状→方法→路线→进度） | 背景→现状→问题目标→内容方法→技术路线→进度→结语 | 只依据 `proposal_spec` + 真实文献 |

非实验型不跑实验引擎：`run_task` 写一份带 `paper_type` 标记的 `03_experiment.json` 桩，
闸门 G01（实验有效硬门）按注册表 `requires_experiment` 自动豁免；G04 零编造硬门仍生效——
综述只引真实文献，设计/案例/报告只依据你给的素材，素材没给的数字一律不写。

```bash
# 文献综述：给主题即可，不用数据集
python auto_research.py plan --topic "基于机器学习的心血管疾病风险预测研究进展" \
    --paper-type survey --out out_survey

# 系统设计：在 spec 里给 system_spec（技术栈/需求/架构/模块/验证）
python auto_research.py run --spec my_design_spec.json --paper-type system_design --out out_design

# 任意素材型类型：把素材写在单独 JSON 里，用 --spec-extra 挂载
#   {"material": {"subject": "...", "background": "...", ...}}
python auto_research.py plan --topic "县域中学课后服务质量提升路径" \
    --paper-type case_study --out out_case --spec-extra my_case.json
```

**加一种新论文类型**只改两处：`paper/thesis_template.py` 的 `PAPER_TYPES` 登记章节骨架，
`paper/paper_type_specs.py` 的 `TYPE_SPECS` 登记素材来源/分章提示/兜底文案。
统一构建器 `_build_non_experimental`、闸门、评审都会自动适配，不用再写 builder。
逐类型自检：`python verify_multitype.py`（加 `--eval` 连评审官一起跑，
加 `--rounds 2` 让**评审官抓到编造**时带意见重写一轮再复评，取分高的一版）。

**写之前先查素材够不够**：`python paper/spec_check.py --spec task_spec.json`
会列出缺哪些字段、哪些偏薄，结果同时落 `00_spec_check.json`。零编造意味着
素材少一块，正文就少一块实质内容——先补齐再跑，比写完再改省事。

**评审意见会回写到正文**：评审官抓出的编造写进 `04_review.json`，下一轮按章节
分发给对应的写作批次重写，再复评取最优；最优若不是最后一轮，正文/评审/docx
一起回滚。契约与 `auto_research.py run` 的重写闭环是同一个。


运行结果写入 `--out`，主要包括 `03_experiment.json`、`plots/`、`06_paper/paper_full.md` 和（安装 `python-docx` 时）`paper_full.docx`。结果是否有效应以实验 JSON 中的指标、错误及跳过方法为准。

## 依赖及数据

项目不是零依赖。`requirements.txt` 包含 OpenAI 客户端、python-docx、NumPy、scikit-learn、SciPy、pandas 和 matplotlib；分析主链依赖 NumPy、scikit-learn、SciPy、pandas、matplotlib。XGBoost、LightGBM、PyTorch 是可选后端。

支持 sklearn 数据集名、本地 CSV/Parquet/Excel/Feather，以及 OpenML（需要联网）。本地文件路径优先相对当前工作目录解析，找不到时再相对仓库根目录。

本地表格文件**支持混合类型**：数值列做中位数插补 + 标准化，字符串类别列做众数插补 + 独热展开（取值超过 20 个的高基数列自动降级为序数编码），所有统计量只在训练折上拟合以避免数据泄漏。预处理明细会随结果写入 `03_experiment.json` 的 `preprocessing` 字段，并在论文第 4 节逐列披露。

```json
{"data": {"source": "csv", "path": "data/my_table.csv", "target": "label"}}
```候选方法可用注册表字符串或完整导入路径字典；`"<seed>"` 可替换为重复实验种子。聚类分支独立使用内置 KMeans/Agglomerative/DBSCAN 实现，默认按 ARI 评估，不执行分类/回归的五种子配对检验。

spec 示例及字段限制见 [SPEC_SCHEMA.md](SPEC_SCHEMA.md)。工程现状、能力边界与逐轮修复记录见 [docs/PROGRESS.md](docs/PROGRESS.md)；通用化演进路线与工程量评估见 [docs/ROADMAP.md](docs/ROADMAP.md)。

系统现在能做到什么、哪些还做不到，见 [docs/CAPABILITY.md](docs/CAPABILITY.md)（能力现状盘点）；
系统为什么这样设计、每条决策否掉了哪些替代方案、失败过在哪里，见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)（答辩用）；补一个新领域后端要改哪四处，见 [docs/BACKENDS.md](docs/BACKENDS.md)；闸门分级与人工放行留痕，见 [docs/GATES.md](docs/GATES.md)；把本地仓库与 GitHub 连起来并自动同步，见 [docs/GITHUB_SYNC.md](docs/GITHUB_SYNC.md)。

请勿将 API Key 写入版本控制；`.env` 本地配置已忽略，`.env.example` 仅提供配置模板。
