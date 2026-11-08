# 本地工程 ↔ GitHub 同步（以及不联网也能交付的备选）

目标：**在这边改完（提交），那边立刻拿到**。同步逻辑封装在
`scripts/github_sync.py`，不依赖额外服务，也不需要装 `gh` CLI。

## 如果不想跟 GitHub 授权纠缠：直接拎包

`python scripts/make_share_package.py` 会生成
`dist/AutoResearch_share_<日期>.zip`（2026-11-09 实测 12.6MB / 120 个文件）：
完整代码、公开数据集、全部文档、三篇已过 G04 的可交付论文（含 Word 与图）、
`start_web.py` 一键启动脚本、`.env.example`、中文使用说明。

排除 `.git/`、`out_*/`、密钥、日志、`memory/` 本地台账与半成品。
**已解压到干净目录实测通过**：网站 200、9 个数据集选项、
三篇示例 `hard_pass 98.3 / 94.8 / 99.2`、图像后端能重新训练、契约校验通过。
拿到包的人不需要 GitHub、不需要账号、不需要 API Key 就能打开网站看成果；
自己跑新论文才需要填 Key。

这条路和 GitHub 同步互不冲突：包随时能生成，GitHub 授权哪天方便哪天补。

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

本机 Git Credential Manager 2.9.0 在**无界面会话里弹不出可用的授权窗口**。
实测三次：默认 browser OAuth（`PUSH_EXIT=124`，300 秒超时）、强制
`credential.gitHubAuthModes=device`（不吐设备码）、`GIT_TERMINAL_PROMPT=0`
非交互探测（同样挂起）；`cmdkey /list` 里始终没有 github.com 条目。
所以自动化这边能做的都做完了，**认证这一步必须本人做一次**。

### 生成令牌：逐屏操作（全程在 GitHub 网页，不用装任何软件）

1. 打开 `https://github.com/settings/tokens`。找不到就直接点这个链接；
   或者走菜单：右上角**头像 → Settings → 左侧最底部 Developer settings →
   Personal access tokens → Tokens (classic)**。
2. 点 **Generate new token (classic)**（不是那个带 "(fine-grained)" 的）。
   可能会让你再输一次账号密码。
3. **Note** 随便填，比如 `AutoResearch`；**Expiration** 选 `7 days`
   （够用，过期自动失效更安全）。
4. 往下找到勾选区，**只勾 `repo` 这一大项**（第一项，Full control of private
   repositories），它下面的小项会自动带上。别的一个都不用勾。
5. 拉到页面最底，点 **Generate token**。
6. 页面会出现一串以 `ghp_` 开头的字符，**只显示这一次**，点旁边的复制按钮。
7. 直接粘到对话框发给我，就这一步，后面不用你管。

> 如果推送报 `403`，通常是没勾 `repo`、令牌过期，或者你在这个仓库只是协作者
> 但邀请还没接受——重新生成一个、或先在邀请页点 Accept 即可。
> 令牌不要发到群里、不要截图公开；用完在**同一个网页**点 `Revoke` 立刻作废。

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
>
> 不想把 token 贴出来也行：自己在**交互式终端**里跑
> `python scripts/github_sync.py`，GCM 会在你自己的桌面上弹 GitHub 授权页，
> 点一次 Authorize 后凭据同样进 Windows 凭据管理器；之后回来跑
> `python scripts/github_sync.py --status` 核对「落后 领先」为 `0 0` 即可。

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
