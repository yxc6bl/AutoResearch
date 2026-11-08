# 本地工程 ↔ GitHub 同步

目标：**在这边改完（提交），那边立刻拿到**。同步逻辑封装在
`scripts/github_sync.py`，不依赖额外服务，也不需要装 `gh` CLI。

## 当前状态（2026-11-08）

| 项 | 状态 |
|---|---|
| 仓库 | `https://github.com/yxc6bl/AutoResearch.git`（用户给的是 `/invitations` 邀请页，已还原成仓库根地址） |
| `origin` | **已配置并 fetch 成功** |
| 远端默认分支 | `main`（本地已由 `master` 改名为 `main`） |
| 本地 HEAD | `f0315eb` —— 合并远端初始快照后的提交，待推送 |
| 网络 | 通（`git ls-remote` 匿名可读，远端 `435d25a`） |
| 推送凭据 | **尚未完成授权**；系统凭据管理器里没有 github.com 条目 |
| 自动推送 hook | 已安装 `.git/hooks/post-commit` |
| 仓库体积 | ~52MB（含 MNIST 15MB），`out_*/`、`.env`、`*.docx`、`corpus/` 已被 `.gitignore` 排除 |

### 远端历史怎么接的

远端是网页上传的旧版快照（两个 `Add files via upload`，含 `AutoResearch.zip`、
`out_cross_v3/` 等），与本地 40+ 次提交**没有共同祖先**。直接推会被拒绝。

处理方式是 `git merge -s ours --allow-unrelated-histories origin/main`：

- 远端两个提交成为祖先，推送是正常 fast-forward（**没有用 force push**）；
- 工作树保持本地最新工程，旧 zip / 旧结果目录不会覆盖新代码；
- 旧快照仍留在历史里，随时可以 `git show` 回溯。

## 授权：当前唯一待办

本机 Git Credential Manager 2.9.0 在**无界面会话里弹不出可用的授权窗口**，
设备码模式也没有吐出码（已实测：browser / device 两种都挂起，系统凭据为空）。
所以最稳的是在 GitHub 网页生成一次 PAT，让我把它存进系统凭据管理器。

**你只需在网页做三步：**

1. 打开 `https://github.com/settings/tokens`（GitHub 网页，Settings →
   Developer settings → Personal access tokens → Tokens (classic)）。
2. Generate new token (classic)，勾 **`repo`**，过期时间随意，生成后复制那串
   `ghp_...`。
3. 把它贴给我。

我这边会执行：

```bash
GITHUB_TOKEN=ghp_xxx python scripts/github_sync.py --store-token
python scripts/github_sync.py            # 首次推送，并验证远端 main 与本地一致
```

`--store-token` 通过 `git credential approve` 把凭据交给 GCM 存进
**Windows 凭据管理器**：不写项目文件、不写 `.git/config`、不进日志、不入库。
之后普通 `git push` 和 post-commit hook 都免密，你也不用再管它。
用完随时可以在同一个网页 Revoke 掉。

> 另一种一次性的用法：`GITHUB_TOKEN=ghp_xxx python scripts/github_sync.py`，
> token 只用 `git -c http.extraheader=...` 临时传给本次 push 进程，
> 进程结束就没了；但这样 hook 的自动推送仍然没有凭据，所以推荐 `--store-token`。
>
> SSH 也可以（生成密钥对、把公钥贴到 GitHub 网页），但国内网络到
> `github.com:22` 常被挡，HTTPS + PAT 是已验证可走通的路径。

## 一次性配置（已完成，留档）

```bash
python scripts/github_sync.py --setup https://github.com/yxc6bl/AutoResearch.git
```

这条命令做四件事：

1. 配置 / 替换 `origin`（若 URL 里带了 token，会先剥掉再写入，避免明文落在 `.git/config`）；
2. 仓库级设置 `credential.helper=manager`（不动全局 git 配置）；
3. 安装 `.git/hooks/post-commit`：**此后每次 commit 自动 push**；
4. 打印结果，可用 `--status` 复核。

不想自动推就加 `--no-hook`，之后手动跑同步命令。

## 日常使用

```bash
python scripts/github_sync.py                 # add -A + commit + push
python scripts/github_sync.py -m "修了某处"    # 指定提交信息
python scripts/github_sync.py --status         # 看 origin / 领先落后 / 未提交改动
```

装了 hook 之后，普通 `git commit` 也会自动推。hook 只推不提交，不会递归触发自己；
推送失败不影响 commit 成功，错误写进 `.git/sync.log`（`.git` 本身不入库）。

## 推什么、不推什么

会推：代码、文档、`data/` 下的公开数据集（Cora 节点表/边表、混凝土 CSV、
购物篮 CSV、MNIST IDX 文件等）。

不推（`.gitignore`）：所有 `out_*/` 运行产物（论文、评分、图表）、`.env`
及任何密钥、`*.docx`、`corpus/` 下载的范文语料、`__pycache__/`、`*.log`。
所以远端拿到的是**可复现的工程**，不是几十份跑出来的半成品。

## 出问题怎么查

```bash
python scripts/github_sync.py --status
cat .git/sync.log | tail -20
git ls-remote origin HEAD       # 单纯测连通性，不需要凭据
```

常见情况：

- `remote: Repository not found` → 地址写错，或私有仓库没授权；
- `Authentication failed` → 系统凭据里没有 github.com 条目，重跑 `--store-token`；
- `non-fast-forward` → 远端有新提交，先 `git fetch origin main && git merge origin/main`；
- hook 没生效 → 检查 `.git/hooks/post-commit` 是否存在、里面 python 路径是否正确，
  重跑一次 `--setup` 即可；
- 想关掉自动推送 → 直接删 `.git/hooks/post-commit` 文件。

## 想更自动

现在已经是「提交即推」。如果还想要「不提交也定时同步」，可以再加一个
定时自动化，每隔若干分钟跑一次 `python scripts/github_sync.py`（无改动时会跳过
commit、只尝试 push）。这是可选项，需要时再说。
