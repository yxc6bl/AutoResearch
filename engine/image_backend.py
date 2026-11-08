#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图像分类后端（source=image）——独立实现，由 analysis_engine.run_analysis 早返回调用。

为什么必须独立分支：主流程的 load_dataset 把 X 包成 DataFrame 交给
build_tabular_preprocessor()，图像是 (N,C,H,W) 张量，走不了那条路；
而且这里要自己管 batch、epoch、参数量与训练耗时，最后再折算回
ANALYSIS_JSON 契约（论文与评审侧只认字段，不认具体模态）。

诚实口径（写进产物，供事实清单核对）：
  - 数据用 MNIST 官方 train/test IDX 文件，本地 gzip，不联网、不造数据；
  - 为控制 CPU 预算，按固定种子分层抽样出子集，抽样规模写进 preprocessing；
  - 预处理只有 /255，没有中位数插补 / 标准化 / 独热编码；
  - 固定随机种子 42 单次训练，报告点估计，不做多种子方差与配对显著性检验；
  - 模型从头训练，没有预训练权重，也没有 GPU；
  - 增广接口已预留但默认关闭，本次实验未使用。
"""
from __future__ import annotations

import gzip
import json
import os
import struct
import time
from pathlib import Path

import numpy as np

IMAGE_MODELS = ("小型卷积神经网络", "多层感知机（图像）", "像素逻辑回归")


# ── 数据加载 ──────────────────────────────────────────────────────────────
def _idx_images(path: Path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path.name} 不是 IDX 图像文件（magic={magic}）")
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    if buf.size != n * rows * cols:
        raise ValueError(f"{path.name} 字节数与头部不符：{buf.size} != {n*rows*cols}")
    return buf.reshape(n, 1, rows, cols), (rows, cols)


def _idx_labels(path: Path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path.name} 不是 IDX 标签文件（magic={magic}）")
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    if buf.size != n:
        raise ValueError(f"{path.name} 标签数与头部不符")
    return buf.astype(np.int64)


def load_idx(root) -> dict:
    """读 MNIST 的四个 IDX gzip 文件，返回 uint8 图像与标签。"""
    root = Path(root)
    Xtr, shape = _idx_images(root / "train-images-idx3-ubyte.gz")
    ytr = _idx_labels(root / "train-labels-idx1-ubyte.gz")
    Xte, shape2 = _idx_images(root / "t10k-images-idx3-ubyte.gz")
    yte = _idx_labels(root / "t10k-labels-idx1-ubyte.gz")
    if shape != shape2 or Xtr.shape[0] != ytr.shape[0] or Xte.shape[0] != yte.shape[0]:
        raise ValueError("MNIST 四个文件的维度互相对不上")
    return {"X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte,
            "shape": shape, "n_train_full": int(Xtr.shape[0]),
            "n_test_full": int(Xte.shape[0])}


def _ensure_cjk_font() -> None:
    """独立运行时也要有中文字体，否则图里的标题全是方框。

    analysis_engine 模块级会注册字体，但这个模块可以被直接调用
    （python -m engine.image_backend），那时没人替它做过这件事。
    """
    import matplotlib
    import matplotlib.font_manager as fm
    matplotlib.use("Agg", force=False)
    for cand in (os.environ.get("CJK_FONT_PATH", ""),
                 "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
        if cand and os.path.exists(cand):
            try:
                fm.fontManager.addfont(cand)
                name = fm.FontProperties(fname=cand).get_name()
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue


def _stratified_subset(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """分层抽样，保持各类比例；返回升序索引，保证可复现。"""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    take = {c: max(1, int(round(n * cnt / len(y)))) for c, cnt in zip(classes, counts)}
    idx = []
    for c in classes:
        pool = np.flatnonzero(y == c)
        idx.extend(rng.choice(pool, size=min(take[c], len(pool)), replace=False).tolist())
    return np.sort(np.asarray(idx, dtype=np.int64))


# ── 模型 ──────────────────────────────────────────────────────────────────
def _build_models(n_classes: int):
    import torch
    import torch.nn as nn

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28 -> 14
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
                nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
                nn.Linear(64, n_classes))

        def forward(self, x):
            return self.net(x)

    class FlatMLP(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(), nn.Linear(dim, 128), nn.ReLU(),
                nn.Linear(128, n_classes))

        def forward(self, x):
            return self.net(x)

    return SmallCNN, FlatMLP


def _random_shift(xb, max_shift: int = 2):
    """训练时随机平移（零填充）。roll 会把边缘绕回来，所以要把绕回部分清零。"""
    import torch
    dx = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    dy = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    if dx == 0 and dy == 0:
        return xb
    h, w = xb.shape[-2], xb.shape[-1]
    out = torch.roll(xb, shifts=(dy, dx), dims=(-2, -1))
    if dx > 0:
        out[..., :dx] = 0
    elif dx < 0:
        out[..., w + dx:] = 0
    if dy > 0:
        out[..., :dy, :] = 0
    elif dy < 0:
        out[..., h + dy:, :] = 0
    return out


def _train_torch(model, Xtr, ytr, Xte, yte, epochs, batch, seed,
                 augment=False, lr=1e-3):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xte_t = torch.from_numpy(Xte).float().div_(255.0)
    yte_t = torch.from_numpy(yte).long()
    history = []
    t0 = time.time()
    Xtr_t = torch.from_numpy(Xtr).float().div_(255.0)
    ytr_t = torch.from_numpy(ytr).long()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(perm), batch):
            sel = perm[i:i + batch]
            xb = Xtr_t[sel]
            if augment:
                xb = _random_shift(xb)
            yb = ytr_t[sel]
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(sel)
        model.eval()
        with torch.no_grad():
            acc = float((model(Xte_t).argmax(1) == yte_t).float().mean().item())
        history.append({"epoch": ep + 1,
                        "train_loss": round(total / len(perm), 4),
                        "test_accuracy": round(acc, 4)})
    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    return pred, history, elapsed


# ── 主流程 ────────────────────────────────────────────────────────────────
def run_image(spec: dict, out_path: Path) -> dict:
    """图像分类：三个模型 × 固定划分 × 固定种子，产出 ANALYSIS_JSON。"""
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    from sklearn.linear_model import LogisticRegression

    d = spec.get("data") or {}
    try:
        ds = load_idx(d.get("root"))
    except Exception as e:  # noqa: BLE001
        return {"error": f"无法加载图像数据：{e}"}

    seed = int(d.get("seed", 42))
    n_train = int(d.get("n_train", 8000))
    n_test = int(d.get("n_test", 2000))
    epochs = int(d.get("epochs", 3))
    batch = int(d.get("batch_size", 128))
    augment = bool(d.get("augment", False))
    n_classes = int(np.max(ds["y_train"]) + 1)

    tr = _stratified_subset(ds["y_train"], n_train, seed)
    te = _stratified_subset(ds["y_test"], n_test, seed)
    Xtr, ytr = ds["X_train"][tr], ds["y_train"][tr]
    Xte, yte = ds["X_test"][te], ds["y_test"][te]

    import torch
    # 多线程下卷积的浮点归约顺序不确定，同一题两次跑会差 0.002。
    # 验收清单要求同题可复现，宁可慢一点也固定成单线程。
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True
    CNN, MLP = _build_models(n_classes)
    _, h, w = Xtr.shape[1], ds["shape"][0], ds["shape"][1]

    results: dict[str, dict] = {}
    histories: dict[str, list] = {}
    preds: dict[str, np.ndarray] = {}
    t_all = time.time()

    # 1) 小型 CNN
    try:
        cnn = CNN()
        p, hist, dt = _train_torch(cnn, Xtr, ytr, Xte, yte, epochs, batch, seed,
                                   augment=augment)
        preds["小型卷积神经网络"] = p
        histories["小型卷积神经网络"] = hist
        results["小型卷积神经网络"] = {
            "params": int(sum(q.numel() for q in cnn.parameters())),
            "framework": "torch", "epochs": epochs, "train_s": round(dt, 2),
            "augment": augment,
        }
    except Exception as e:  # noqa: BLE001
        results["小型卷积神经网络"] = {"error": str(e)}

    # 2) torch 多层感知机（扁平像素）
    try:
        mlp = MLP(h * w)
        p, hist, dt = _train_torch(mlp, Xtr, ytr, Xte, yte, epochs, batch, seed,
                                   augment=augment, lr=1e-3)
        preds["多层感知机（图像）"] = p
        histories["多层感知机（图像）"] = hist
        results["多层感知机（图像）"] = {
            "params": int(sum(q.numel() for q in mlp.parameters())),
            "framework": "torch", "epochs": epochs, "train_s": round(dt, 2),
            "augment": augment,
        }
    except Exception as e:  # noqa: BLE001
        results["多层感知机（图像）"] = {"error": str(e)}

    # 3) 传统基线：扁平像素上的逻辑回归（无 epoch、无增广）
    try:
        t0 = time.time()
        flat_tr = Xtr.reshape(len(Xtr), -1).astype(np.float32) / 255.0
        flat_te = Xte.reshape(len(Xte), -1).astype(np.float32) / 255.0
        lr = LogisticRegression(max_iter=200, random_state=seed)
        lr.fit(flat_tr, ytr)
        p = lr.predict(flat_te)
        preds["像素逻辑回归"] = p
        results["像素逻辑回归"] = {
            "params": int(np.prod(lr.coef_.shape) + lr.intercept_.size),
            "framework": "sklearn", "epochs": None, "train_s": round(time.time() - t0, 2),
            "augment": False, "solver": str(lr.solver), "max_iter": int(lr.max_iter),
        }
    except Exception as e:  # noqa: BLE001
        results["像素逻辑回归"] = {"error": str(e)}

    if not preds:
        return {"error": "所有图像模型都失败", "details": results}

    metrics = []
    for name, pred in preds.items():
        acc = float(accuracy_score(yte, pred))
        f1 = float(f1_score(yte, pred, average="macro", zero_division=0))
        info = results[name]
        results[name].update({"accuracy": round(acc, 4), "f1_macro": round(f1, 4)})
        metrics.append({
            "name": name, "value": round(acc, 4),
            "setting": (f"参数量={info.get('params')}；epoch={info.get('epochs') or '不适用'}；"
                        f"训练耗时={info.get('train_s')}s；macro-F1={f1:.4f}"),
        })
    best = max(preds, key=lambda k: results[k]["accuracy"])
    y_pred = preds[best]
    cm = confusion_matrix(yte, y_pred, labels=list(range(n_classes))).tolist()
    rep = classification_report(yte, y_pred, output_dict=True, zero_division=0)
    per_class = {}
    for c in range(n_classes):
        r = rep.get(str(c), {})
        per_class[f"class_{c}_precision"] = round(float(r.get("precision", 0)), 4)
        per_class[f"class_{c}_recall"] = round(float(r.get("recall", 0)), 4)
        per_class[f"class_{c}_f1"] = round(float(r.get("f1-score", 0)), 4)
    total_s = round(time.time() - t_all, 2)

    # ── 三张图 ────────────────────────────────────────────────────────────
    plots_dir = out_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots = []
    try:
        _ensure_cjk_font()
        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings("ignore", message="Glyph .* missing from font")
        names_t = [n for n in histories if n in preds]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for n in names_t:
            hst = histories[n]
            axes[0].plot([x["epoch"] for x in hst], [x["train_loss"] for x in hst],
                         marker="o", label=n)
            axes[1].plot([x["epoch"] for x in hst], [x["test_accuracy"] for x in hst],
                         marker="o", label=n)
        axes[0].set_title("训练损失"); axes[1].set_title("测试集准确率")
        for ax in axes:
            ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(str(plots_dir / "image_training.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        plots.append({"title": "训练损失与测试准确率随 epoch 变化",
                       "caption": "图 5-1 训练损失与测试准确率随 epoch 变化",
                       "file": "image_training.png"})

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(np.asarray(cm), cmap="Blues")
        ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
        ax.set_title(f"混淆矩阵（{best}）")
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=7,
                        color="red" if cm[i][j] > np.max(cm) * 0.5 else "black")
        fig.tight_layout()
        fig.savefig(str(plots_dir / "image_confusion.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        plots.append({"title": f"混淆矩阵（{best}）",
                       "caption": f"图 5-2 混淆矩阵（{best}）",
                       "file": "image_confusion.png"})

        fig, axes = plt.subplots(4, 4, figsize=(7, 7))
        for k, ax in enumerate(axes.ravel()):
            ax.imshow(Xte[k, 0], cmap="gray")
            ok = int(y_pred[k]) == int(yte[k])
            ax.set_title(f"真{yte[k]}/预{y_pred[k]}", fontsize=8,
                         color="green" if ok else "red")
            ax.axis("off")
        fig.suptitle("测试集样例与预测（前 16 张）")
        fig.tight_layout()
        fig.savefig(str(plots_dir / "image_samples.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        plots.append({"title": "测试集样例与预测",
                       "caption": "图 5-3 测试集样例与预测（前 16 张）",
                       "file": "image_samples.png"})
    except Exception as e:  # noqa: BLE001
        print(f"[image] 绘图失败：{e}")

    image_facts = {
        "dataset": d.get("label"), "format": "IDX (MNIST gzip)",
        "n_train_full": ds["n_train_full"], "n_test_full": ds["n_test_full"],
        "n_train_used": int(len(Xtr)), "n_test_used": int(len(Xte)),
        "input_shape": [1, int(ds["shape"][0]), int(ds["shape"][1])],
        "n_classes": n_classes, "classes": [str(i) for i in range(n_classes)],
        "seed": seed, "epochs": epochs, "batch_size": batch,
        "normalization": "像素值除以 255", "augmentation": "未使用（接口已预留，默认关闭）",
        "device": "cpu", "pretrained": False,
        "torch_version": torch.__version__,
        "total_s": total_s,
        "models": {k: v for k, v in results.items()},
        "history": histories,
        "best": {"name": best, "accuracy": results[best]["accuracy"],
                 "f1_macro": results[best]["f1_macro"]},
    }

    return {
        "task_type": "classification",
        "best_method": best,
        "best_value": results[best]["accuracy"],
        "best_metric": "accuracy",
        "metrics": metrics,
        "per_class": {best: per_class},
        "confusion_matrix": {best: cm},
        "significance": ("未做配对显著性检验：本实验在固定划分与固定随机种子下各训练一次，"
                         "报告的是点估计，不声称差异具有统计显著性。"),
        "plots": plots,
        "skipped_methods": [k for k, v in results.items() if v.get("error")],
        "preprocessing": {
            "n_samples": int(len(Xtr) + len(Xte)),
            "n_raw_features": int(ds["shape"][0] * ds["shape"][1]),
            "note": ("图像张量 (N,1,28,28)；仅做 /255 缩放；不做列级标准化、"
                     "中位数插补或独热编码；按固定种子分层抽样子集。"),
        },
        "image": image_facts,
        "findings": (f"MNIST 官方划分 {ds['n_train_full']}/{ds['n_test_full']} 张，"
                     f"按种子 {seed} 分层抽取 {len(Xtr)}/{len(Xte)} 张；"
                     f"{epochs} 个 epoch、batch={batch}、CPU 从头训练共用 {total_s}s。"
                     f"最优为 {best}，top-1 准确率 {results[best]['accuracy']:.4f}、"
                     f"macro-F1 {results[best]['f1_macro']:.4f}。"),
        "adaptations": [],
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="图像分类后端自检")
    ap.add_argument("--root", default="data/mnist")
    ap.add_argument("--out", default="out_image_probe")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()
    spec = {"task_type": "classification", "metric": "accuracy",
            "data": {"source": "image", "root": a.root, "label": "MNIST 手写数字图像数据集",
                     "n_train": a.n_train, "n_test": a.n_test, "epochs": a.epochs,
                     "batch_size": 128, "seed": 42}}
    out = Path(a.out); out.mkdir(exist_ok=True)
    res = run_image(spec, out / "03_experiment.json")
    print(json.dumps({k: res.get(k) for k in
                      ("error", "best_method", "best_value", "metrics", "plots")},
                     ensure_ascii=False, indent=2)[:1500])
