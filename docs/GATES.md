# 闸门与权限表

这份文档回答三个问题：这次运行到底卡在哪、谁能批准放行、凭什么说系统架构是成立的。
它同时是工程说明和论文「系统设计」章的证据来源，不写没有对应产物的设想。

## 一、为什么单独做一层

检查点原本散在四个地方：`mark_delivery()` 判降级稿、`eval_agent` 判零编造硬门、
`literature_review_agent` 判文献覆盖、`trace` 记执行过程。每处都能给出结论，但没有一张表能回答
「这次能不能交」「是谁放行的」。参考系统论文里的权限边界设计，把这四处收成一份
`07_gate_report.json`，每条闸门带**依据产物、实测数值、来源、把握程度**。

关键约束：**这一层不新增限制**。`engine/gates.py` 的 `CATALOG` 只登记代码里已有的判定，
状态一律从真实产物读取，读不到就记 `unknown`，绝不猜成 `pass` 或 `fail`。
分类只影响报告与交付裁决，不改变各阶段行为。

## 二、三级与放行权限

| 级别 | 含义 | 严格模式下的效果 | 谁能放行 |
|---|---|---|---|
| P0 | 不可交付（真实性、篇幅、模型参与、硬门） | `fail` 或 `unknown` 都阻塞，退出码 3 | 系统拒绝；人工可留痕放行，但原始证据不抹除 |
| P1 | 明显缺陷（评分门、文献池） | 只提示，不阻塞 | 导师或作者留痕放行 |
| P2 | 信息性（覆盖率、审评、引用、评审介入、轨迹） | 只记录，用于复盘 | 无需放行 |

P0 的 `unknown` 同样阻塞，理由是「不能证明达标」而不是「已判不合格」。
`--no-strict` 只用于查看，会把 `unknown` 从阻塞里去掉，`fail` 依旧阻塞。

## 三、闸门清单

| 编号 | 级别 | 阶段 | 检查什么 | 依据产物 | 不通过时的处置 | 放行权限 |
|---|---|---|---|---|---|---|
| G01 | P0 | 实验 | 无 `error`、有 `metrics` 与 `best_method` | `03_experiment.json` | 实验失败不得成文，先修数据/方法绑定 | 系统拒绝；人工放行需说明数据来源 |
| G02 | P0 | 写作 | 正文汉字 ≥ 6000（与 `mark_delivery` 同口径，不是新指标） | `06_paper/paper_full.md` | 模板骨架稿不可交付，等模型恢复重跑写作 | 系统拒绝；人工放行需书面确认 |
| G03 | P0 | 写作 | `deliverable=true` 且 `llm_written≠False` | `06_paper/DELIVERY_STATUS.json` | 确定性降级稿只用于自测/定位链路 | 系统拒绝；人工放行需书面确认 |
| G04 | P0 | 评审 | `gate=hard_pass`、`consistency=100`、评审官未发现幻觉 | `06_paper/07_eval_verdict.json` | 结论与裁决表矛盾或编造数字，必须回滚改写 | 系统拒绝，不接受放行 |
| G05 | P1 | 评审 | `passed=true` 且 `overall≥60` | `07_eval_verdict.json` | 按问题清单补强后重评 | 导师或作者留痕放行 |
| G06 | P1 | 文献 | `papers` 非空（**不设固定篇数下限**） | `01_literature.json` | 篇数由题目需求与角色覆盖决定，按缺口补抓 | 作者留痕放行 |
| G07 | P2 | 文献 | 带 `llm_what`/`llm_fit` 的文献比例 | `01_literature.json` | 0% 说明本轮纯关键词筛选，报告如实标低可信 | 无需放行 |
| G08 | P2 | 文献 | 覆盖度、是否 LM 审评、缺口主题 | `04_literature_review.json` | 缺口写进下轮检索式 | 无需放行 |
| G09 | P2 | 写作 | `n_cited>0`，入选文献真的进正文 | `05_lit_feedback.json` | 检查第 2 章引用块是否落进正文 | 无需放行 |
| G10 | P2 | 评审 | `llm_judge.available=true` | `07_eval_verdict.json` | 网关挂时只有规则评分，必须标明覆盖面不足 | 无需放行 |
| G11 | P2 | 轨迹 | 有阶段记录且无 `error` 状态 | `00_trace.jsonl` | 看 `trace` 定位失败阶段与耗时 | 无需放行 |

新增闸门必须同时给出产物路径和判定依据，否则不许进 `CATALOG`。

## 四、人工放行怎么留痕

```
python auto_research.py gates --out out_lit_final --approve G03,G05 --who 杨心程 --why "自测确认模板稿仅用于链路定位"
```

`--who` 与 `--why` 必填，不接受空口放行。记录写进 `07_gate_approvals.json`（只增不改），
报告里对应闸门标 `waived`，同时保留 `raw_status` 原始证据；P0 的放行会在
`decision.human_override` 里单独列出。也就是说，放行表示「人做了决定」，
不是「系统把问题改好了」，答辩时能直接指到人和理由。

脚本/CI 用法：`python auto_research.py gates --out DIR --exit-code`，不可交付时返回 3。

## 五、来源与把握程度

反馈台账按 `human > model > rule` 排序，避免把模型意见当成硬事实：

- 规则评分门的 issue：`rule/high`
- LLM 评审官的幻觉、无支撑断言：`model/medium`
- 文献审评的覆盖缺口：`model/medium`（LM 成功）或 `rule/medium`（网关不可用的规则兜底）
- 人工批注：由 `--provenance` / `--confidence` 指定
- 文献逐篇筛选：`01_literature.json` 里每篇带 `confidence`，取值 `lm_read > lm_semantic > keyword_only`

## 六、一次实测：分数高不等于能交

对 `out_lit_final` 跑 `gates`，结果很能说明为什么要这层：

```
[P0][未通过] G02 正文仅 4655 个汉字
[P0][未验证] G03 未写 DELIVERY_STATUS.json
[P1][通过]   G05 passed=true，overall=91.0，等级 优秀
[P2][未通过] G07 0/17 篇带 LM 阅读判断：本轮纯关键词兜底
[P2][未通过] G10 LLM 评审官未介入：本轮只有规则评分
结论：不可交付 —— 阻塞 G02；未验证 G03
```

规则评分门给了 91 分「优秀」，但那是**在降级稿上打的分**：正文不到 5000 字、
文献没逐篇读、评审官没介入。单看 `overall` 会误判为高质量交付。
交付裁决必须看 P0 的硬事实，分数只作为 P1 的质量信号。

## 七、真实失败模式（架构章可直接引用的证据）

以下每条都实际发生过，括号里是修复提交，能复跑对应命令验证，不是设想出来的风险。

| 现象 | 根因 | 修复 | 留下的证据 |
|---|---|---|---|
| 网关挂时一次探测卡 60 秒以上 | SDK 内部重试与本模块退避叠乘 | `b705c71`：`max_retries=0`、探测 6 秒快速失败 | `python auto_research.py doctor` |
| 低质降级稿被当成成品 | 写作失败仍返回 0、只打印提示 | `d6813f0`：默认严格模式，退出码 2/3，产物标【不可交付】 | `06_paper/DELIVERY_STATUS.json` |
| 入口查了网关，中途写挂仍返回 0 | `from_title` 缺少收尾裁决 | 本轮补 `_strict_gate_exit()`，两个入口共用 | `gates --exit-code` |
| 严格模式判定根本没执行 | `cmd_run` 里 `out_dir` 未定义，`NameError` | 本轮改为 `Path(res['out_dir'])` | 自测 `_strict_gate_exit(...) == 3` |
| 错配题目白跑几分钟才发现 | 没有预演 | `39a8da9`：`from_title --dry-run` | 预演打印数据集/方法/检索式 |
| 4.8 万行上 SVC 跑 8 分钟 | RBF 核 O(n²) | `20058fa`：大样本自动换线性/直方图法，实测 35 秒 | `03_experiment.json` 的 `adaptations` |
| 文献凑满 45 篇却大量跑题 | 固定数量 + 关键词命中 | `6117189`、`1c11283`、`ef86a19`：角色覆盖、三档召回、LM 逐篇读摘要 | `01_literature.json` 的 `llm_what` |
| 主流程评审意见没进台账 | 只有 `step eval` 会采集 | 本轮在 `run_task` 按最终 verdict 记一次 | `memory/feedback.jsonl` |
| 说不清哪个阶段慢/挂了 | 只有产物没有过程记录 | `b4adc4f`：`00_trace.jsonl` | `python auto_research.py trace --out DIR` |

## 八、架构论证四段骨架

1. **问题分解**：论文生成的四个硬约束是证据不可信（编造数字）、过程不可追溯（答辩问不出处）、
   跨学科没数据、模型服务不可靠。每条都对应本系统一个产物：
   `07_eval_verdict.json` / `00_trace.jsonl` / `03_experiment.json` / `DELIVERY_STATUS.json`。
2. **取舍**：选固定流水线而非自由 agent 循环，因为同题两次跑必须可复现、能指到随机种子与产物；
   选文件契约而非内存消息，因为断点续跑和子 Agent 单独调用都要跨进程；
   硬门只保留四条 P0，其余降级为提示，避免把标准写死成僵化规则。
3. **一致性**：六个模块能否拼成闭环——`task_spec` → `01_literature` → `02_hypotheses` →
   `03_experiment` → `06_paper` → `07_eval_verdict`，加上 `00_trace`、`feedback.jsonl`、
   `07_gate_report` 三个横向台账；重跑复用已有产物，`--force` 才重抓文献。
4. **边界**：不做什么（人文类全自动成文、图表/时序后端）、哪些尚未验证（见下节）。

## 九、未验证清单（不许写成已完成）

- LM 逐篇读摘要、LM 文献审评、LLM 评审官这三条链路依赖网关，网关恢复前只验证了规则兜底路径。
  恢复后按序验证：`doctor` → `from_title --force`（看 G07/G08/G10 是否都通过）→
  `gates --out DIR --json`。
- 「同题生成稿 vs 真实优秀范文」的质量 benchmark 仍缺范文全文与同口径材料；
  现有的 `benchmark_titles.py` 是**可执行性** benchmark（33 道真实高校题中仅 21% 能匹配本地数据集），
  两者不是一回事。
- 中文库（知网等）导入路径、用户自传证据包的整合计算，仍在设计阶段。
