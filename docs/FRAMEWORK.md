# AutoResearch 框架与共享记忆契约（v1）

> 本文档是"子代理拆分 + 个性化记忆"的总设计。它定义了所有子代理共用的**共享状态（记忆库）**与**模块契约**。
> 目的：让本工程从"写死的线性流水线"升级为"多子代理通过记忆库协作"，并让负责 Memory / 个性化的组员有明确接口可接。

---

## 1. 为什么需要这份框架

当前 `auto_research.py` 是一条写死的线性流水线：写 spec → 抓文献 → 跑实验 → 出论文，**跑完即弃，跨次运行无状态**，模块间没有统一黑板。

这带来两个问题：
1. 无法"拆分子代理"——各阶段硬编码在同一个函数里，改一处容易崩全局。
2. 无法"记忆 / 个性化"——每次都从零抓文献、不记得使用者偏好，写作风格也无从蒸馏。

因此第一步不是直接写子代理，而是先定**记忆库 + 模块契约**这块骨架。骨头定下，子代理和组员的 Memory 模块才能各自往里接。

---

## 2. 总体架构

```
                        ┌─────────────────────┐
                        │   Orchestrator      │  （编排器，原 auto_research.py 重构）
                        │   负责路由与调度     │
                        └──────────┬──────────┘
                                   │ 通过 MemoryStore 读写
              ┌────────────────────┼────────────────────┐
              │                    │                     │
   ┌──────────▼───┐      ┌─────────▼────────┐    ┌───────▼────────┐
   │ Literature   │      │   PlanningAgent  │    │  ExperimentAgent│
   │ Agent        │      │   (planner.py)   │    │ (analysis_engine)│
   └──────────┬───┘      └─────────┬────────┘    └───────┬────────┘
              │                    │                     │
              └─────────┬──────────┴─────────┬───────────┘
                        │                    │
                 ┌──────▼──────┐      ┌──────▼──────┐
                 │ WritingAgent│      │  EvalAgent  │
                 │(generate_   │      │(audit_consistency│
                 │  paper.py)  │      │  扩展为评分门) │
                 └──────┬──────┘      └─────────────┘
                        │
                 ┌──────▼──────┐
                 │ MemoryAgent │  ← 组员负责（个性化 + 记忆）
                 │ (lit/user/  │     读取 lit_memory + 使用者行为
                 │  style 画像)│     写回 user_profile / style_profile
                 └─────────────┘

        所有代理都通过 MemoryStore 共享状态，不直接互相调用。
```

---

## 3. MemoryStore（记忆库）契约

记忆库是全部代理共用的"黑板"。先定义接口与记录类型，**实现由组员在 Memory 模块完成**，我们这边先定契约。

### 3.1 接口（伪代码）

```
MemoryStore.get(namespace, key)            -> record | None
MemoryStore.put(namespace, key, record)    -> void
MemoryStore.query(namespace, filter)       -> list[record]
MemoryStore.list(namespace)                -> list[key]
```

命名空间：`literature/` 、`user/` 、`style/`

### 3.2 记录类型一：lit_memory（文献记忆）

见 `memory/schema/lit_memory.schema.json`。核心字段：

```
{
  "papers": [
    {
      "id": "arxiv:2312.12345",        // 去重主键（arXiv id / DOI / URL）
      "title": "...",
      "authors": "...",
      "venue": "arXiv:cs.LG",
      "year": "2023",
      "url": "https://arxiv.org/abs/2312.12345",
      "abstract": "...",                // 截断至 ~700 字
      "fetched_at": "2026-10-31T17:20:00",
      "source_query": "random forest diabetes",
      "cited_in": ["out_single_v3/round_1"]
    }
  ]
}
```

去重规则：以 `id` 为主键；同一 `id` 再次出现时合并 `cited_in`，不重复写入。

### 3.3 记录类型二：user_profile（使用者偏好画像）—— 组员实现

```
{
  "lit_pref": {                      // 偏好哪类文献
    "topic_weights": {"医疗诊断": 0.6, "集成方法": 0.4},
    "preferred_venues": ["arXiv", "中文核心"],
    "preferred_methods": ["随机森林", "SVM"]
  },
  "writing_pref": { ... }            // 偏好哪类论证/表达（待组员细化）
}
```

### 3.4 记录类型三：style_profile（成文习惯 / nano缪斯）—— 组员实现

```
{
  "sentence_patterns": [...],        // 常用句式
  "section_rhythm": {...},           // 章节节奏
  "vocab": [...],                    // 用词习惯
  "anti_ai_markers": [...]          // 用于反 AI 腔的特征
}
```

`style_profile` 由 WritingAgent 在成文时读取，产出"像那个人自己写的"文本。

---

## 4. 各子代理 I/O 契约

| 代理 | 输入 | 输出 | 负责方 |
|------|------|------|--------|
| LiteratureAgent | `query, max_results` | 增量写入 `lit_memory` | 我们（先改 literature.py 接 MemoryStore） |
| PlanningAgent | `topic / query` | `spec` JSON | 我们（planner.py 封装） |
| ExperimentAgent | `spec` | `03_experiment.json`（真实数字） | 我们（analysis_engine.py 封装） |
| WritingAgent | `spec + analysis + style_profile` | `paper`（md/docx） | 我们（generate_paper.py 接 style_profile） |
| EvalAgent | `paper + analysis` | `verdict`（评分门） | 我们（audit_consistency.py 升级） |
| MemoryAgent | `lit_memory + 使用者行为` | `user_profile / style_profile` | **组员** |

所有代理通过 MemoryStore 传递状态；Orchestrator 只做调度，不内联业务逻辑。

---

## 5. 阶段分工与顺序

| 阶段 | 内容 | 负责方 |
|------|------|--------|
| A 基础 | 仓库落地 + 本框架契约 + MemoryStore 接口 + 编排器改走 MemoryStore | 我们 |
| B 子代理拆分 | 先 LiteratureAgent → Planning/Experiment/Writing 模块化 | 我们 |
| C 能力 | 文献扩源（中文源）+ 评价 agent（评分门） | 我们 |
| D 记忆+个性化 | lit_memory 持久化 → 偏好建模 → 风格蒸馏（nano缪斯） | **组员** |
| E 交付 | GitHub 远端 + 网页界面 | 我们 |

---

## 6. 第一步落地（当前）

1. 仓库已从压缩包落地为正式工程并初始化 git（见根目录）。
2. 本文件 + `memory/schema/lit_memory.schema.json` 即"框架契约"，组员可据此接 MemoryAgent。
3. **已实现**：`memory/store.py` 提供 MemoryStore 参考实现（get/put/query/list + 四命名空间，本地 JSON）；`agents/literature_agent.py` 封装 `run()`，流程为"先查 literature/ 命名空间去重 → 抓 arXiv → 按 id 去重写回 MemoryStore（合并 cited_in、记 fetched_at/source_query）→ 写 `01_literature.json` 兼容论文生成器"。`auto_research.py` 三处文献调用已切换为 LiteratureAgent。文献记忆真正落地，为组员偏好建模提供数据底座。

> 演进说明：MemoryStore 是"接口契约的参考实现"。组员后续可替换为数据库 / 向量库后端，只要四个方法签名与语义不变，各子代理无需改动（只加不改）。
