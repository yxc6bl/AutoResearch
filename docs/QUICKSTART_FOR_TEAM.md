# 给组员：连自己的 API 跑一次（约 5 分钟准备）

这个包是 2026-11-11 整理后的最新版，已合并两边工程、修掉两篇问题论文暴露的缺陷，
并把评审改成「范文参照 + 专家质量诊断 + feedback 回流」。

## 1. 准备

```bash
pip install -r requirements.txt
cp .env.example .env          # Windows 用 copy .env.example .env
```

编辑 `.env`，填**你自己的**三行（地址、Key、模型名要配套）：

```
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=你的Key
OPENAI_MODEL=gpt-4o-mini
LLM_EXTRA_BODY={}
```

Kimi / Gemini / DeepSeek / 通义 / 智谱 / 本地 Ollama 的地址表见 `.env.example`。
也可以不写 `.env`，直接开网页，在左栏「模型接口」选预设、填 Key 运行。

## 2. 先确认模型真能用（这一步不能跳）

```bash
python auto_research.py doctor
```

必须看到「模型可用」。若显示不可用，**不要继续跑**——跑出来只会是模板骨架稿
（`WRITE_STATUS.json` 里 `llm_written=false`），语义不通、中英混杂，闸门会判不可交付。

## 3. 跑一次完整流程

```bash
python auto_research.py plan --topic "你的研究主题" --paper-type survey --out out_team
```

想看网页就 `python web/app.py`，打开 `http://127.0.0.1:5000`，
在「最近产物」里点 `out_team`，七个栏目都能看。

## 4. 重点看这三个地方

| 文件 | 看什么 |
|---|---|
| `06_paper/WRITE_STATUS.json` | `llm_written` 必须是 `true`；是 `false` 说明 Key/地址没生效 |
| `06_paper/08_paper_review.md` | 专家诊断、与范文的实质差距、优先修改清单 |
| `06_paper/07_eval_verdict.json` | `overall` 只是规则合规分；`quality_assessment.status` 才是质量评审是否完成 |

`07_gate_report.json` 的 `decision.deliverable` 为 `true` 才算可交付。

## 5. 反馈怎么留

`memory/feedback.jsonl` 会记下专家意见；对某条表态：

```bash
python auto_research.py lit-pref --review-mark out_team P01 accepted --note "照这个改"
```

采纳/驳回会进记忆，影响下一轮；想让专家意见直接驱动重写，加环境变量
`AR_EXPERT_REVISE=1`。

## 注意

- 包里**不含 `.env`**，不会带走任何人的 Key。
- 包里不含历史运行产物和日志，解压即可开工。
- 两边工程的合并状态与待接线点见 `docs/MERGE_STATUS.md`。
