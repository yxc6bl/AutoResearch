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
| G04 | P0 | 评审 | 评审官**已介入**且未发现幻觉；`gate=hard_pass`、`consistency=100`（缺席记未验证） | `06_paper/07_eval_verdict.json` | 结论与裁决表矛盾或编造数字，必须回滚改写 | 系统拒绝，不接受放行 |
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

### 6.2 网关恢复后的真实全链路（`out_gate_real`，2026-11-05）

`doctor` 显示校园网关恢复（1.7s 探测可用，arxiv / crossref / openalex 三源均可用），
于是用「基于随机森林的乳腺癌分类研究」（569 样本 / 30 特征）跑了一轮 `--force` 全链路。

| 环节 | 实测结果 |
|---|---|
| 文献检索 | 351.9s，250 篇候选 → 45 篇入选（45 是 `top_n` **上限**不是目标；新版本 `selection` 报告会写 `n_eligible` 与 `capped_by_cap`，把「被上限截断」和「自然饱和」区分开） |
| LM 逐篇读摘要 | 28/45 篇（62.2%）带 `llm_what`/`llm_relation`，10 篇由 LM 语义筛选认可，7 篇仍为关键词兜底；28 篇写明「用在哪节」：相关工作 11 / 方法 10 / 实验对比 7 |
| LM 文献审评 | `lm_driven=true`，覆盖度 7.0/10，给出 5 个具体缺口（近三年综述、Transformer/GNN 对比、TCGA 大数据集、超参与解释性、SHAP/LIME）和 10 篇弱相关 |
| LLM 写作 | 正文 16872 个汉字（模板骨架稿只有 4 千多字） |
| 规则评分门 | `overall=99`「优秀」、`passed=true`、`gate=hard_pass` |
| LLM 评审官 | 68 分、`verdict=revise`、12 条疑似幻觉（例如第 3.3.3 节写成 McNemar 检验而真实实验是配对 t 检验；数据无缺失却写「中位数插补」） |
| 闸门裁决 | **不可交付**：`G04` 未通过（幻觉），`G03` 未验证；尽管 `G05` 给了 99 分 |

这一轮的价值不在于「跑通了」，而在于它同时证明了两件事：LM 读文献/审评/写作/评审
四条链路确实都在工作；以及**只看评分门的分数会把幻觉稿当优秀稿交付**。
P0 的硬事实必须独立于加权分。

**第二轮与修复后复测**：评审闭环回滚重写后正文 17064 字，规则评分升到 99.5，
LLM 评审仍 68 分、幻觉未消除；`mark_delivery` 判【可交付】（它只看字数与模型参与），
闸门层仍以 `G04` 拦下，进程退出码 3。两个结论打架的问题已修：闸门裁决会回写到
`DELIVERY_STATUS.json` 的 `gate_decision` 字段，并把 `deliverable` 改一致。

网关随后间歇性恢复，用修好的评审口径单跑 `step eval` 复测：幻觉从 12 条降到 5 条，
且 5 条全是真事实矛盾（McNemar 检验写成配对 t 检验、数据无缺失却写「中位数插补」、
随机种子编号不一致、臆测 p 值、事实清单之外的特征重要性表）；
「题目点名的方法不是最优」「四舍五入」这类假阳性不再被算作幻觉。
规则分仍是 99.5，但 `gate=hard_fail`、`passed=False`，G04 未通过 → 不可交付。

网关再次挂掉时（6.2s 快速失败），评审官缺席会写进 `verdict["llm_judge"]`，
G04 记 `unknown`（未验证）而不是 `pass`，严格模式下同样阻塞交付——
「没审」永远不等于「没幻觉」。

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
| 评审官查出 12 条幻觉，评分门仍判 99 分优秀并通过 | 综合段重算 `hard_ok`，把第 10 步刚设的 `gate=hard_fail`/`passed=False` 覆盖回通过 | 本轮：幻觉拦截并入最终 `hard_ok`，并写 `llm_hallucination_gate`；闸门 G04 双保险（原始清单 + 最终标记） | `out_gate_real/06_paper/07_eval_verdict.json` 与 G04 报告 |
| 回滚重写没拿到幻觉清单 | `04_review.json` 只写了 `unsupported_claims`/`suggestions`，写作端也只读前者 | 本轮：简报补 `hallucinations`，写作端把两类意见合并注入第三批写作提示 | `04_review.json` 与新代码的注入段 |
| 读了摘要却说不清「用在哪节」 | `usable_in` 只被用来推导 `role` 后就丢掉，没写进 `01_literature.json` | 本轮：写 `llm_usable_in` 与每篇 `confidence`，正文引用块标注用途，G07 报告覆盖率 | `01_literature.json` 的字段与 G07 |
| 评审官把「题目点名的方法不是最优」当成编造 | 评审提示没交代忠实选题口径，也没区分四舍五入与编造 | 本轮：提示里明确「题目点名方法只指定研究对象，不要求最优」「舍入一致不算幻觉」「改进建议不写成幻觉」 | 复测幻觉 12 条 → 5 条，且全为真矛盾 |
| 网关挂了、评审官缺席，`verdict` 里连字段都没有，被当成「无幻觉」 | `eval_agent` 只在 `available=True` 时写 `llm_judge` | 本轮：缺席也写 `{"available":false,"reason":...}`；G04 记 `unknown` 并阻塞 | `gates` 单测：缺席 → `unverified=['...','G04']`、不可交付 |
| 「取最优那一轮」只记了分数，稿件早被下一轮覆盖 | 评审闭环没有留每轮快照，也没真正回滚 | 本轮：每轮留 `paper_full_roundN.md` 与分轮 verdict，同分按幻觉数择优，最优非末轮时回滚 md 与 verdict 并写 `REVIEW_ROUNDS.json` | `REVIEW_ROUNDS.json`（docx 仍属末轮，需重转换） |
| `DELIVERY_STATUS` 里 `llm_written` 永远是 null，G03 只能判未验证 | `run_task` 调 `mark_delivery` 没透传写作端的返回值 | 本轮：透传 `llm_written`；并把闸门裁决回写 `gate_decision`，消除「可交付 / 不可交付」两种结论 | `DELIVERY_STATUS.json` 的 `gate_decision` |
| 单独跑 `step eval` 的文献分和主流程对不上 | `step eval` 没把 `01_literature.json` 传给评分门 | 本轮：读同一份精选文献传入，并在输出里显示评审官是否介入 | `step eval` 打印与主流程同口径的综合分 |

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

- LM 逐篇读摘要、LM 文献审评、LLM 写作、LLM 评审官四条链路已在 2026-11-05 实测跑通（见 6.2）。
  幻觉拦截、评审提示口径已用 `step eval` 复测；但「幻觉清单回传写作端后，完整跑一轮能否把那 5 条
  事实矛盾真正改掉」尚未验证——这需要再跑一次约 20 分钟的 `from_title --force`。
  校园网关是间歇可用的，`doctor` 必须每次先跑，不可用时的结论一律是「未验证」而非「通过」。
- `select_by_role` 新加的 `n_eligible` / `capped_by_cap` 尚未在完整链路上实测；
  本轮 `n_selected=45` 只能说明撞到了上限，不能说明是否还有合格候选被舍弃。
- 「同题生成稿 vs 真实优秀范文」的质量 benchmark 已经有一条部分可用的路径：
  `public_benchmark.py` 从 Crossref 抓真实已发表论文题录，与 10 个公开数据集配成 20 对，
  实验层本地跑通（`p01` 5.2s、6 项指标），`compare` 同时给出原文元数据对比与
  `corpus/` 里北大/清华优秀范文的指纹差距。但**还不是全文级同题对比**：Crossref 只给元数据与摘要，
  多数期刊全文有版权墙；而且能同题跑的是应用型期刊论文，与高质量范文在题目复杂度上不在一个量级。
  要补全文，得另抓 arXiv / PMLR / OpenReview 的开放获取 PDF。
- 现有的 `benchmark_titles.py` 是**可执行性** benchmark（33 道真实高校题，重跑结果 15 道可执行 /
  45.5%，早期 21% 是数据集目录只有 7 个时的旧数），与质量 benchmark 不是一回事。

## 十、文献源的现实约束（2026-11-05 实测）

同一次 `doctor` 里：crossref 可用 10 条、arxiv 可用 2 条、openalex 可用 50 条（另一时刻又 429）、
s2 429、europepmc 503、dblp 直接返回反爬挑战页（Anubis，202 + HTML，不是 JSON）。
**换 LLM API 只解决模型通道，完全不会修好这些文献源的限流**——它们是不同的服务商。
所以文献侧的正确做法不是无限加 API，而是：多源冗余 + 单源 429 自动降权（已有）、
填 `LIT_CONTACT_EMAIL` 进 polite pool、以及 `corpus/imports/` 的本地导入源（不联网、永不限流，
也是中文文献与导师指定必引文献的入口）。默认六源全开，按稳定性排序，任一源挂了不影响其余源。

Crossref 的定位要说准：它是 DOI 元数据注册商，**期刊覆盖最全**，部分会议录有 DOI，
但 CS 顶会与预印本远不全覆盖；顶会覆盖本来该靠 dblp（被反爬挡住）、Semantic Scholar、arXiv、OpenAlex。
「Crossref 是抓顶会的主力」这个理解只对一半。

## 十一、文献侧闭环：审评意见要真的接回检索端

此前三个环节是断的：抓+读 → 审评覆盖/缺口 → `04_literature_review.json` 落盘 → 结束。
审评给的 `recommended_add_queries` / `missing_themes` 从来没人再拿去搜；
GatherAgent 内部两轮补抓（缺口回填、覆盖不足扩面）新增的文献**没有经过 LM 逐篇读**，
实测就是 45 篇里只有 28 篇带 `llm_what`，剩下 17 篇纯关键词兜底。

改动：
- 新增 `agents/literature_refetch_agent.py`：审评布尔检索式清洗（`"A" AND ("B" OR "C")` 这种
  语法 Crossref / OpenAlex 并不解析，原样送过去召回极差）→ 补抓 → 按质量分截 36 篇送读 →
  重新按角色覆盖精选 → 覆盖 `01_literature.json`，写 `0x_gap_refetch.json` 留痕。
- GatherAgent 两轮补抓同样补读；最终入选的未读文献再补读一次（⑤-d）。
- 精选上限：`balanced` 45→60、`strict` 25→35、`recall` 60→80，`LIT_TOP_N` 可配；
  报告保留 `n_eligible` / `capped_by_cap`，让「被上限截断」始终可见。

实测（同一批真实候选）：45 篇 → 60 篇，LM 阅读覆盖 **62.2% → 100%**，
60 篇全部带 `llm_usable_in`（写明用在哪节），耗时 170s。
第一次只放宽上限、没做入选补读时，覆盖率反而从 62.2% 掉到 61.7%——
**篇数增加本身不等于质量，必须配套阅读覆盖，否则只是堆噪声**。
想把时间省回来：`LIT_REFETCH=0` 关补抓，或调小 `LIT_TOP_N`。

## 十二、领域拓展：能力到位 ≠ 数据到位

- 时序：`engine/analysis_engine.py` 新增 `timeseries` 数据源。公开数据是墨尔本
  1981-1990 每日最低气温（3650 行，真实 CSV，本地缓存优先、缺失才下载，下载失败明确报错不造数据）。
  滑动窗口 lag 7 + 季节正余弦 + 月份 → 回归预测下一日。实跑 26.9s，梯度提升树 R²=0.5647、6 项指标。
- 聚类：修了预检豁免的 bug（中文名「K均值」被当成未知估算器全部跳过，报「没有可运行的候选方法」）。
  实跑 KMeans ARI=0.6201、4 项指标、2.8s。
- 题库 33 道真实高校题：**18 道可执行（54.5%）**，其中 4 道标 `proxy`——
  建模能力到位但领域数据不同；**同领域真实数据可执行 14 道**，误匹配 0。
  去掉 `无监督`/`簇` 这类泛词后，北大《域迁移与场景重建》不再被误判成可执行的聚类题。

必须说清：交通流题配气温序列只是**时序建模能力基准**，不是真做了交通流；
鸢尾花分群也不是真实业务用户数据。要真正接住这些题，仍需用户自传领域数据，
或引入对应公开数据集。视觉/视频后端、文本与语言模型、图结构、领域数据缺失
仍是硬缺口，没有因为这次改动消失。
- 中文库（知网等）导入路径、用户自传证据包的整合计算，仍在设计阶段。
