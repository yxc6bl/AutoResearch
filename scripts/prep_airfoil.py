"""把 UCI 翼型气动噪声（NASA 风洞真实实验）原始 .dat 整理成标准 CSV。

原始文件：6 列、空格分隔、无表头。
列含义：
  1. frequency           频率 (Hz)
  2. angle_of_attack     攻角 (度)
  3. chord_length        弦长 (m)
  4. free_stream_velocity 来流速度 (m/s)
  5. thickness           吸力面位移厚度 (m)
  6. sound_pressure_db   标度化声压级 (dB)  —— 回归靶值
"""
import csv

SRC = "data/_airfoil.dat"
DST = "data/airfoil_noise.csv"
HEADER = [
    "frequency", "angle_of_attack", "chord_length",
    "free_stream_velocity", "thickness", "sound_pressure_db",
]

rows = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()  # 兼容任意空白分隔
        if len(parts) != 6:
            continue
        rows.append([float(x) for x in parts])

with open(DST, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(HEADER)
    w.writerows(rows)

print(f"wrote {DST}: {len(rows)} rows, {len(HEADER)} cols")
print("target(sound_pressure_db) range:",
      min(r[-1] for r in rows), "~", max(r[-1] for r in rows))
