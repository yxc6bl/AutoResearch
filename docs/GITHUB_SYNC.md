# 本地工程 ↔ GitHub 同步

目标：**在这边改完（提交），那边立刻拿到**。同步逻辑封装在
`scripts/github_sync.py`，不依赖额外服务，也不需要装 `gh` CLI。

## 当前状态（2026-11-08）

| 项 | 状态 |
|---|---|
| 本地仓库 | 已初始化，40+ 次提交，工作区干净，HEAD `51b1ceb` |
| `origin` | **未配置** —— 缺你 GitHub 上那个仓库的地址 |
| 推送凭据 | **未配置** —— 本机有 Git Credential Manager 2.9.0，但没有存过 github.com 凭据 |
| 网络 | 通（`git ls-remote https://github.com/git/git.git` 实测成功） |
| 自动推送 hook | 脚本已写好，**执行 `--setup` 后才安装** |
| 仓库体积 | 37MB / 59 个跟踪文件，最大单文件 148KB；`out_*/`、`.env`、`*.docx`、`corpus/` 已被 `.gitignore` 排除 |

也就是说：代码这边全准备好了，只差**仓库 URL**和**一次授权**。

## 一次性配置

把你在 GitHub 网页上建好的仓库地址贴过来（形如
`https://github.com/<用户名>/<仓库名>.git`），然后：

```bash
python scripts/github_sync.py --setup https://github.com/OWNER/REPO.git
```

这条命令做四件事：

1. 配置 / 替换 `origin`（若 URL 里带了 token，会先剥掉再写入，避免明文落在 `.git/config`）；
2. 仓库级设置 `credential.helper=manager`，用本机 GCM 存凭据（不动全局 git 配置）；
3. 安装 `.git/hooks/post-commit`：**此后每次 commit 自动 push**；
4. 打印结果，可用 `--status` 复核。

不想自动推就加 `--no-hook`，之后手动跑同步命令。

## 授权（三选一，按省事程度排）

**A. 让 GCM 弹窗登录（推荐，一劳永逸）**

```bash
python scripts/github_sync.py
```

首次推送会弹 GitHub 授权窗口，登录一次后凭据存进 Windows 凭据管理器，
以后所有推送都不再问。

**B. Personal Access Token（无图形界面 / 弹窗被挡时用）**

在 GitHub → Settings → Developer settings → Personal access tokens 生成一个
`repo` 权限的 token，然后只在当前会话注入：

```bash
GITHUB_TOKEN=ghp_xxx python scripts/github_sync.py
```

脚本用 `git -c http.extraheader=...` 临时传给 push 进程，**不写 remote URL、
不写 .git/config、不写 hook 文件**，进程结束就没了。

**C. SSH key**（若你习惯 SSH）：自行生成 key、把公钥加到 GitHub，
再用 `git@github.com:OWNER/REPO.git` 跑 `--setup`。脚本不做任何改动。

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
git ls-remote origin HEAD       # 单纯测连通性，不需要本地提交
```

常见情况：

- `remote: Repository not found` → 地址写错，或仓库是私有的但没授权；
- `Authentication failed` → 走上面的 A 或 B；
- hook 没生效 → 检查 `.git/hooks/post-commit` 是否存在、里面 python 路径是否正确，
  重跑一次 `--setup` 即可；
- 想关掉自动推送 → `git config --local --unset credential.helper` 不解决，
  直接删 `.git/hooks/post-commit` 文件。

## 想更自动

现在已经是「提交即推」。如果还想要「不提交也定时同步」，可以再加一个
定时自动化，每隔若干分钟跑一次 `python scripts/github_sync.py`（无改动时会跳过
commit、只尝试 push）。这是可选项，需要时再说。
