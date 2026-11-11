# 两边工程合并状态（2026-11-11）

主干：`yxc6bl/AutoResearch` 的 `main`（本工程）。组员工程包 `AutoResearch-main.zip`
来自另一个分叉，两边各自大改，不能简单覆盖。

## 已并入（无冲突，可直接 import）

| 文件 | 作用 |
|---|---|
| `engine/provenance.py` | 运行清单 `00_run.json`、环境指纹、模型请求快照 |
| `engine/io_utils.py` | 原子写 / 追加写 JSONL |
| `engine/task_spec.py` | task_spec 版本校验 |
| `paper/fact_pack_store.py` | `FACT_PACK.json` 写入与 SHA-256 校验 |
| `docs/decisions/001-auditable-facts-and-runs.md` | 组员的设计决策记录 |
| `extras/windows/*` | Windows 开机自启 / 启停脚本（未设为默认启动方式） |

## 未并入及原因

- `engine/config.py`：要求 Key 长度 ≥20 且所有配置纯 ASCII，会挡住本地/自建端点，
  与「任意 OpenAI 兼容 API 都能接」冲突。本工程由 `engine/llm_client.channels()`
  统一解析配置，已支持多通道与短 Key。
- `web/index_session.html`（聊天式 UI）：把整段输入当 `topic` 提交，曾把类型素材
  JSON 原样印进论文标题，是两篇问题论文的入口。本工程保留七栏工作台，
  并在前后端都加了主题校验；聊天式界面若要回归，必须先补同样的校验。
- `web/app.py.session-ui` / 各类 `.bak`：备份文件，不进主干。

## 需要接线才能生效的部分（下一步）

`FACT_PACK` 与 `00_run.json` 目前只是可用模块，尚未在主链路写盘：

1. `paper/generate_paper.py` 写正文后调用
   `fact_pack_store.write_fact_pack(06_paper/FACT_PACK.json, paper_type=..., facts=...)`，
   并把返回的 sha256 写进 `WRITE_STATUS.json`；
2. `agents/eval_agent.py` 读该 sha256 写入 verdict，实现「写作/评审同源」；
3. `auto_research.py run` 开头 `provenance.start_run_manifest()`、结尾 `finish_run_manifest()`；
4. `engine/llm_client.create_chat()` 调 `provenance.record_llm_call()` 记请求快照。

接线后需重跑全套回归与一次端到端，确认不破坏闸门与零编造护栏。
在此之前，这些模块不影响现有链路，也不参与任何评分。
