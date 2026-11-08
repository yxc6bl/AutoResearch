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

运行结果写入 `--out`，主要包括 `03_experiment.json`、`plots/`、`06_paper/paper_full.md` 和（安装 `python-docx` 时）`paper_full.docx`。结果是否有效应以实验 JSON 中的指标、错误及跳过方法为准。

## 依赖及数据

项目不是零依赖。`requirements.txt` 包含 OpenAI 客户端、python-docx、NumPy、scikit-learn、SciPy、pandas 和 matplotlib；分析主链依赖 NumPy、scikit-learn、SciPy、pandas、matplotlib。XGBoost、LightGBM、PyTorch 是可选后端。

支持 sklearn 数据集名、本地 CSV/Parquet/Excel/Feather，以及 OpenML（需要联网）。本地文件路径优先相对当前工作目录解析，找不到时再相对仓库根目录。

本地表格文件**支持混合类型**：数值列做中位数插补 + 标准化，字符串类别列做众数插补 + 独热展开（取值超过 20 个的高基数列自动降级为序数编码），所有统计量只在训练折上拟合以避免数据泄漏。预处理明细会随结果写入 `03_experiment.json` 的 `preprocessing` 字段，并在论文第 4 节逐列披露。

```json
{"data": {"source": "csv", "path": "data/my_table.csv", "target": "label"}}
```候选方法可用注册表字符串或完整导入路径字典；`"<seed>"` 可替换为重复实验种子。聚类分支独立使用内置 KMeans/Agglomerative/DBSCAN 实现，默认按 ARI 评估，不执行分类/回归的五种子配对检验。

spec 示例及字段限制见 [SPEC_SCHEMA.md](SPEC_SCHEMA.md)。工程现状、能力边界与逐轮修复记录见 [docs/PROGRESS.md](docs/PROGRESS.md)；通用化演进路线与工程量评估见 [docs/ROADMAP.md](docs/ROADMAP.md)。

系统为什么这样设计、每条决策否掉了哪些替代方案、失败过在哪里，见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)（答辩用）；补一个新领域后端要改哪四处，见 [docs/BACKENDS.md](docs/BACKENDS.md)；闸门分级与人工放行留痕，见 [docs/GATES.md](docs/GATES.md)；把本地仓库与 GitHub 连起来并自动同步，见 [docs/GITHUB_SYNC.md](docs/GITHUB_SYNC.md)。

请勿将 API Key 写入版本控制；`.env` 本地配置已忽略，`.env.example` 仅提供配置模板。
