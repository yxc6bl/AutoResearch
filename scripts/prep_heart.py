#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清洗 UCI Heart Disease（processed.cleveland.data）为带表头、目标二值化的标准 CSV。

来源（公开、无需登录）：
  https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

原始 14 列无表头，部分字段含 "?" 缺失标记（ca/thal 列），最后一列为诊断
（0=无心血管疾病，1~4=有）。本脚本：加标准表头 + 把 "?" 置空（交给引擎插补）
+ 最后一列二值化为 target（0/1），便于直接喂给 AutoResearch。
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "_heart_raw.data"
OUT = ROOT / "data" / "heart_disease.csv"

COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def main() -> None:
    rows = []
    for line in RAW.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        if len(parts) != 14:
            continue
        # "?" 缺失标记置空，交给引擎插补
        clean = ["" if p == "?" else p for p in parts]
        # 最后一列二值化：>0 表示存在心血管疾病（原值 1~4）
        tgt = clean[-1]
        clean[-1] = "1" if tgt not in ("", "0") else "0"
        rows.append(clean)

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)

    pos = sum(1 for r in rows if r[-1] == "1")
    print(f"写入 {OUT}：共 {len(rows)} 样本，正类(患病)={pos}，负类={len(rows) - pos}")


if __name__ == "__main__":
    main()
