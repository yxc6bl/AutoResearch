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

## 网页界面

零第三方依赖（只用标准库），起一个本地服务就能用：

```bash
python web/app.py                                      # 默认 http://0.0.0.0:5000
WEB_HOST=127.0.0.1 WEB_PORT=5077 python web/app.py    # 本机换端口
```

左侧填主题 / 类型 / API Key 直接跑；右侧分七栏看结果：**论文、文本诊断、评分、
质量闸门、文献、实验、图表**。看历史成果不需要 Key——点「最近产物」里任意一条，
或填目录名即可加载，`.docx` 与 `.md` 一键下载。

- 深链：`/?dir=out_verify_survey&tab=gate`、`/?demo=1`、`/?theme=dark`，可直接定位到
  某个产物的某一栏（截图、发给导师都方便）。
- 「人工选材」在文献栏内展开：勾选保留 / 剔除并写备注，提交即写入
  `memory/lit_preferences.json`，下一轮抓取生效，台账同步标 `by=human`。
- 运行日志按阶段推进（解析 → 检索 → 实验 → 写作 → 评审 → 闸门 → 导出），
  中途失败会停在出错阶段并把原因留在日志里。
- 服务端只读工程根内的产物目录（`_safe_dir`），`/api/file` 也做了路径越权拦截。

## 模型网关挂了怎么办

`doctor` 报「模型不可用」、写作退回模板骨架、闸门 G03 判不可交付，都属于**同一个外部故障**：
网关上游没响应。先确认是谁的锅（不要猜）：

```bash
python auto_research.py doctor
curl -i https://apiai.sztu.edu.cn/v1/models
```

响应里出现 `Server: APISIX/...`、`X-APISIX-Upstream-Status: 502`，就是网关自己的上游挂了；
同时访问 `https://api.github.com` 若正常，说明本机网络、证书、Key 都没问题。这种情况下
换一个可用的 OpenAI 兼容端点就能恢复，代码不用改。文献源是另一套服务，不受影响。

**方式一：直接换主通道**——改 `.env` 的三行 `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`。
新进程第一次握手就走新端点。

**方式二：加一条备用通道**——在 `.env` 里按 `.env.example` 补 `_2` 后缀的四行，保留原网关。
主通道连续失败 3 次会自动切过去；`available()` 探测也会逐条试完所有通道，不会只探主通道
就判定「不可用」而把写作端按死在模板兜底。两条都挂才走确定性降级。

换完执行 `python auto_research.py doctor` 确认「模型可用」，再重跑一次流水线拿可交付成稿。
三个注意点：端点必须兼容 `/v1/chat/completions`；多数商用端点不认
`chat_template_kwargs.thinking`，备用通道的 `LLM_EXTRA_BODY_2` 留 `{}`；模型要能稳定输出
万字中文和 JSON（评审 Agent 走 `chat_json`），小模型会让评审链路退化。

## 接任意 AI 接口（GPT / Kimi / Gemini …）

系统**不绑定任何厂商**。底层走标准 OpenAI SDK，只要求对方提供 `/v1/chat/completions`
兼容协议；预设列表只是帮你把地址填好，地址和模型名随时能改。

**网页端**：左栏「模型接口」选预设（OpenAI / DeepSeek / Kimi / Gemini / 通义 / 智谱 /
硅基流动 / 本地 Ollama / 自定义），地址、模型名、额外请求体会自动填好，三项都可手改；
再填**对应厂商**的 Key 就能跑。配置只注入本次子进程的环境变量，不落盘、不写日志；
浏览器只记住地址、模型和预设，不记 Key。高级参数里还能指定命名风格。

**命令行 / `.env`**：改三行即可，完整地址表见 `.env.example`：

```bash
OPENAI_BASE_URL=https://api.moonshot.cn/v1
OPENAI_API_KEY=你的_Kimi_Key
OPENAI_MODEL=moonshot-v1-8k
LLM_EXTRA_BODY={}
```

三个容易踩的坑，系统已经替你处理：

- **地址和 Key 必须配套**。早期版本网页只开放 Key、服务端把地址写死成学校网关，
  换任何厂商都会失败；现在配置按你填的值注入，`.env` 只在留空时兜底。
- **额外请求体**。学校网关这类 vLLM 系才需要 `chat_template_kwargs.thinking=false`；
  换到别家留 `{}`。网页选预设会自动填；就算忘了改，端点报「不认识的参数」时
  兼容层也会自动丢掉它重试。
- **推理模型参数名**。o1 / o3 / GPT-5 不收 `temperature`，要用
  `max_completion_tokens`；`LLM_PARAM_STYLE=auto` 会按模型名自动判断，
  判断错了还能根据 400 报错自动降级。规划、写作、评审三处调用共用这一套适配。

**边界**：覆盖所有 OpenAI 兼容端点（含 Gemini 的 `/v1beta/openai/` 兼容层、各类中转站、
本地 Ollama）。如果某家只给私有协议（例如 Anthropic 原生 `/messages`），需要另写一层
适配器；把它的接口文档给我，我按同一套通道机制接进来，不影响现有链路。

## 专家式论文质量评审（不是模板检查）

评分门 `eval_agent` 只回答「有没有」：零编造、篇幅、章节、引用编号、格式。
它产出的 `overall` 现在明确标为 **`rule_compliance`（规则合规分）**，
并在 `quality_assessment` 里注明是否做过模型评审——模板骨架拿 93 分
不再被当成「论文写得好」。真正判断产出质量的是 `paper_review_agent`：

- **参照系来自高质量范文**：`benchmark_corpus.reference_profile()` 从 `corpus/`
  的北大 2024 本科毕设等范文抽取质量标尺（篇幅、章节、文献量、原创性、
  消融/跨数据集/显著性、论证连接词密度、图表与公式引用），注入评审 prompt。
  PDF 提取失败的乱码样本会自动剔除；往 `corpus/` 放新的中文 `.txt` 即可扩展。
  这是把好论文整理成评阅标尺，**不是微调模型**。
- **专家只管「好不好」**：通读全文，判断论证是否成立、证据是否支撑结论、
  文献是否真被用起来、有无独立见解、文字是否可读；明确要求点名模板拼接稿
  （缺主语、中英混杂、研究问题塞进摘要、文献流水账）。输出不给分数、不给等级。
- **范文对标在网关不可用时也有**：`quality_gap()` 给出可核对的确定性差距，
  诊断书里照样能看，不会只剩一句「未完成诊断」。
- **feedback 闭环**：专家的 `top_actions` 按影响排序写进 `memory/feedback.jsonl`
  （`type=专家评审`，带为什么改、改哪一章），并挂到 `04_review.json` 的
  `expert_actions`；设 `AR_EXPERT_REVISE=1` 时这些意见会并入下一轮写作提示。
  默认不自动重写——实测「只有风格建议」直接重写容易退步。
  人也能逐条表态：`lit-pref --review-mark <目录> P01 accepted --note "…"`。

网页「文本诊断」栏展示诊断书，「评分」栏顶部固定提示「规则分≠论文质量」。

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

**文献摘要接入是可选实验能力**：`01_literature.json` 里带 abstract 的文献，其摘要
节选本来可以进入写作事实包与评审官事实基础（两边逐字同源）。但四轮实测都没能超过
零幻觉基线——模型一旦看到摘要里的样本量，就会给没有依据的文献也补数字。因此
**默认关闭**，设 `AR_LIT_ABSTRACTS=1` 才启用；实测与结论见 `docs/PROGRESS.md` 第 18 节。
这轮真正修掉的是引用块里的 `gist_cn` 数字一直没进评审事实基础，导致真实素材被误判编造。

**文献一句话概要是两端同源的素材**：检索阶段落盘的 `gist_cn` 现在同时进入
写作事实包与评审事实基础，模型写「研究现状」不再只看题名就替文献编结论。
占位值（「未知」「无」「N/A」等）会被确定性清掉，不会写进正文引用块。

## 选材留痕、人工偏好与文本诊断

**文献侧**：每次抓取写 `LIT_SELECTION.json`，记录候选池里每一篇的去留、理由和
决策者（模型／规则／人工）；用 `python auto_research.py lit-pref --selection <产物目录>`
查看。人的偏好存进 `memory/lit_preferences.json`，**单条立刻生效**，下轮抓取就会
剔除或加权，文献审评 Agent 也读同一份偏好，不会把你排除的类型又建议补回来：

```bash
python auto_research.py lit-pref --kind avoid --field keyword --value preprint --note "不要预印本"
python auto_research.py lit-pref --drop "题名或 DOI" --note "与题目无关"
python auto_research.py lit-pref --keep "题名或 DOI" --note "导师指定保留"
python auto_research.py lit-pref --list
```

**论文侧**：整篇产出后自动写 `06_paper/08_paper_review.md`（给人读）和同名 json。
它**不给分数**，只做指导教师式的通读诊断：整体判断、逐章问题（引用原文片段／
为什么是问题／具体怎么改）、按影响排序的优先修改清单。也可单独跑
`python auto_research.py step --name review --out <产物目录>`。每条意见有稳定编号，
可以对它表态，采纳或驳回都会存进记忆并影响下一轮写作提示：

```bash
python auto_research.py lit-pref --review-mark <产物目录> P01 accepted --note "照这个改"
python auto_research.py lit-pref --review-mark <产物目录> P02 rejected --note "这条不适用"
```

网关不可用时诊断书会明确标注「未完成文本诊断」，只列确定性事实，不假装模型读过全文。

**检索规划也留痕**：拿选题后判断「要抓什么、走哪条路径」这一步现在写成
`LIT_PLAN.json`——每条检索式的用途与来源、启用的数据源、每轮（首轮／扩面／
缺口补抓）实际用了哪些式子抓回多少、最终入选多少。用途是确定性词面推断，
字段里明确标了来源，不冒充模型的推理过程。

**网页上就能选**：加载任一产物目录后，切到「文献」标签，点「人工选材」，
会列出 AI 入选清单并标注每篇状态（AI 入选／人工保留／已剔除）。取消勾选即剔除，
可填备注；保存后写进偏好记忆与本次选材台账（标 `by=human`），下次抓取生效。
旧目录没有检索计划时会明确提示，不会显示成空白。

**系统能拿反馈改自己**：论文诊断和文献审评指出的「缺什么文献」会转成英文检索式
存进记忆，下一轮抓取自动并入检索计划并标 `origin=feedback`；诊断书里有一节
「本次检索已应用的历史反馈」，能看到闭环确实转起来了。同一条反馈只消费一次。

**改代码后跑一次回归**（64 个用例，不联网、约 0.3 秒）：

```bash
python -m unittest discover -s tests -v
```

覆盖零编造边界、写作端与评审端事实基础同源、gist 素材真实送达写作提示、占位清理、主题一致性预警、降级恢复与构建超时、
重写意见分发、G03 判据不循环、素材完备度与网关负缓存过期。


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
