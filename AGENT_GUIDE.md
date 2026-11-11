# AGENT_GUIDE —— 给操作本工程的 AI 助手

## 你的角色与目标

你拿到的是 **AutoResearch**：一个通用自动化科研引擎。它不接受「写死答案」，而是**接收一份研究规格（spec），自动跑真实的机器学习实验，并产出一篇结构完整、数字真实的中文论文**。

你的任务**不是复现某个既定结果**，而是：

1. 帮用户在 Docker（或本地）里把引擎正确运行起来；
2. 准备一份合理的研究规格（spec）；
3. 执行命令、读取产物、按下方「质量自检清单」判断产出是否够好；
4. 若不够好，调整 spec 或环境后重跑，**让工具自己计算，而不是替它编造结论**。

## 三条铁律（务必遵守）

1. **零编造**：所有指标、表格、图表都来自引擎真实运行。你与论文生成器都不得伪造数字、文献或结论。
2. **让工具跑，而不是替它写结果**：你负责准备 spec、执行命令、核查产物；实验与写作由引擎完成。
3. **质量由自检清单判定**，不由「是否和某篇样例一致」判定。不要预设「它应该跑出某指标」。

## 工程结构（精简可运行版）

```
AutoResearch/
├── auto_research.py          # 统一入口：run / plan / iterate 三个子命令
├── planner.py                # 把研究主题转成合法 spec（需要 LLM Key，无 Key 降级模板）
├── literature.py             # 用 arXiv 免费 API 检索真实文献（无需 Key）
├── engine/analysis_engine.py # 确定性分析引擎
├── paper/generate_paper.py   # 论文生成器（消费流水线产物）
├── engine/
│   ├── analysis_engine.py
│   └── tasks/proof/          # 示例 spec（wine / 回归 / 聚类 / 多数据集 / 混合后端）
├── paper/
│   └── generate_paper.py
├── SPEC_SCHEMA.md            # spec 字段契约（写 spec 前必读）
├── README.md                 # 工程总览
├── requirements.txt          # 运行依赖；XGBoost/LightGBM/PyTorch 可选
├── Dockerfile / docker-compose.yml / .dockerignore
└── .env.example              # 复制为 .env 并填入你自己的 API Key
```

## 三种用法

入口只有一个：`python auto_research.py <子命令>`。

**1）run —— 已知 spec，直接跑**
```
python auto_research.py run --spec engine/tasks/proof/wine_sklearn_spec.json --out ./out_task
# 只跑分析不出论文：追加 --no-paper
```
产物落在 `--out` 目录：`01_literature.json`、`02_hypotheses.json`、`03_experiment.json`、`06_paper/paper_full.md` 与 `.docx`、若干 `.png` 图。

**2）plan —— 只给主题，让 Planner 生成 spec 再跑**
```
python auto_research.py plan --topic "比较随机森林与梯度提升在结构化数据上的表现" --out ./out_plan
```
Planner（任意 OpenAI 兼容端点）把主题扩写成合法 spec；若无 Key 则降级为模板 spec。文献检索走 arXiv 免费接口，不依赖 Key。

**3）iterate —— 让引擎自己迭代深化**
```
python auto_research.py iterate --topic "比较不同分类器在结构化数据集上的表现" --out ./out_iter --rounds 3
```
每轮把上一轮真实结果喂回 Planner，自动「扩宽数据集 / 增加方法 / 收紧子假设」后重跑，最多 N 轮。适合证明「换数据、换方法仍稳健」。

## Docker（可选）

仓库包含 Docker 配置，但本地 Python 命令是主要运行方式；容器依赖和命令以 Dockerfile/docker-compose.yml 为准。

**A. 准备密钥**（不要把 Key 烤进镜像）
```
cp .env.example .env
# 编辑 .env，填入你自己的 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
```

**B. 构建镜像**
```
# 基础版（仅 sklearn 后端，镜像最精简）
docker build -t autoresearch:latest .

# 进阶版（一并安装 xgboost / lightgbm / torch 后端）
docker build -t autoresearch:extras --build-arg EXTRAS=1 .
```
说明：进阶后端「装了就用，不装则引擎自动跳过该方法并记录在 `skipped_methods`」，不影响主链路。

**C. 运行**
```
# 方式一：docker compose（已挂载 .env 与产出目录、可挂自己的 spec）
docker compose up --build
# 想跑自己的 spec：在 docker-compose.yml 取消注释 command 行，把 spec 放到 ./my_specs 后挂载为只读

# 方式二：直接 docker run
docker run --rm -v "$PWD/.env":/app/.env -v "$PWD/out_docker":/app/out_docker \
  autoresearch:latest run --spec engine/tasks/proof/wine_sklearn_spec.json --out /app/out_docker
```

**D. 取回产物**：容器内 `/app/out_docker` 已挂载到宿主 `./out_docker`，论文与图表在 `out_docker/06_paper/`。

## 如何写一份能跑出好结果的 spec

spec 契约见 `SPEC_SCHEMA.md`。让结果更好的几条经验（不是硬性规则）：

- **数据集**：用 `sklearn` 内置名（如 `wine`、`breast_cancer`、`digits`）最稳；自有数据用 `csv`/`parquet`/`excel`；开放数据用 `openml` id。多数据集对比在 spec 里写 `datasets: [...]`。
- **方法**：`candidate_methods` 用字典式写完整 import 路径，例如
  `{"estimator": "sklearn.ensemble.RandomForestClassifier", "params": {"n_estimators": 300, "random_state": "<seed>"}}`。
  `<seed>` 占位符会被引擎替换为每轮随机种子，保证可复现。方法越多样（线性 + 树 + 集成），结论越有说服力。
- **指标**：分类默认 `accuracy`（objective maximize），也可 `f1_macro`；回归用 `r2`/`mse`。
- **子假设**：`sub_hypotheses` 写具体可验证的机制命题（如「集成模型比线性模型训练-测试 gap 更小」），引擎会自动抽取真实数字给出裁决 `verdict`。

## 质量自检清单（判断「跑得好不好」）

跑完后，**你应当打开产物逐项核对**，而不是只看「有没有生成文件」：

- [ ] `03_experiment.json` 中 `best_method` 非空、`best_value` 合理（分类 accuracy 显著高于随机基线，回归 r2 明显大于 0）。
- [ ] `skipped_methods` 符合预期：装了 xgboost 还被跳过 → 环境/依赖有问题，需排查；没装而被跳过属正常。
- [ ] `seed_variance`（多种子方差）数值小，说明方法稳定、结果可信。
- [ ] `significance` 字段有结论（配对显著性检验通过与否）。
- [ ] `06_paper/paper_full.md` 行数充足（通常 >100 行）、含摘要/引言/相关工作/方法/实验/讨论/结论等完整章节、表格与图片引用非空。
- [ ] `06_paper/` 下 `.png` 图表成功生成，且中文显示正常（无方框）。
- [ ] 多数据集任务：存在 `cross_dataset.png` 与跨数据集对比章节、排名稳定性表。
- [ ] 文献：`01_literature.json` 含真实可访问的 arXiv 条目（标题/作者/年份/URL）。

若任一项不达标，按「常见问题」排查后调整 spec 重跑。**达标即可交付，不必与任何样例数字对齐。**

## 常见问题

- **图表中文变方框**：本地需系统有中文字体；Docker 镜像已装 `fonts-noto-cjk`，容器内不会此问题。
- **论文生成器报 API 错**：检查 `.env` 的 `OPENAI_API_KEY/BASE_URL/MODEL`；即使 LLM 不可用，生成器会降级为确定性骨架，论文仍能产出（只是文字非 LLM 润色）。
- **某方法被跳过**：通常是该后端未安装（xgboost/lightgbm/torch），装好即可；或 spec 路径写错。
- **Python 版本**：建议使用 Python 3.11，并通过 `python -m pip install -r requirements.txt` 安装实际依赖。

## 你不该做的事

- 不要预先写好 `best_method`、指标数字或结论再「让工具去验证」——这是作弊，违背了本工程「真实计算」的根基。
- 不要把 `out_*` 历史产物当答案直接交给用户；每次都是工具重新跑出的。
- 不要为「凑好看」去改动 `analysis_engine.py` 的核心计算逻辑；若发现评测维度不足，应通过扩展 spec 或新增方法解决。
