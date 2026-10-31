#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计生成的论文：裁决表 与 结论 是否一致；是否出现「素材未提供」式谎言。

用法：
  python audit_consistency.py <paper_full.md路径>

退出码 0 = 通过；1 = 发现矛盾/谎言。
"""
import re
import sys
from pathlib import Path


def parse_verdict_table(md: str):
    """从 markdown 中提取子假设裁决表，返回 {H1: 裁决, ...}。"""
    rows = {}
    # 匹配 | 子假设 | 裁决 | 依据 | 表头之后的数据行
    in_table = False
    for line in md.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 3:
            continue
        if cells[0] in ("子假设", "假设", "子假设/裁决") or "裁决" in cells[1]:
            in_table = True
            continue
        if in_table:
            first = cells[0]
            m = re.match(r"^(H\d)", first)
            if m:
                hid = m.group(1)
                rows[hid] = cells[1]
    return rows


def parse_conclusion(md: str):
    """从结论段提取每个 H 的裁决：形如 H1（证据不足）：...

    注意：正文里「结论」二字还可能出现在局限性等小节（如「结论向更大规模…」），
    因此必须锚定以 ## 开头的结论标题，而非简单 rfind("结论")。
    """
    concl = {}
    # 找最后一个「以 # 开头、含『结论』二字的标题」，从该标题之后开始解析
    heads = list(re.finditer(r'^#{1,4}\s+.*结论.*$', md, re.M))
    if heads:
        idx = heads[-1].start()
    else:
        idx = md.rfind("结论")
    seg = md[idx:] if idx >= 0 else md
    for m in re.finditer(r"(H\d)\s*[（(]\s*([^）)]+?)\s*[）)]", seg):
        hid, v = m.group(1), m.group(2).strip()
        concl[hid] = v
    return concl


def detect_lies(md: str):
    """检测真正的「素材谎言」：把单数据集 §5.1 错标成跨数据集表并声称缺失，
    或在明明有数据的地方声称「素材未提供」。

    注意：单数据集研究中诚实声明「未采集逐类明细 / 未做配对显著性检验」是正当的，
    不计入谎言。只抓把单数据集 §5.1 错标为「跨数据集对比表」并声称缺失的误导表述。
    """
    flags = []
    # 1) 把 §5.1 错标为跨数据集对比表并声称缺失（单数据集研究中危害最大）
    mislabel = re.search(r"素材未提供[^。\n]{0,20}第\s*5\.1\s*节跨数据集对比表", md)
    if mislabel:
        s = max(0, mislabel.start() - 20)
        flags.append("错位引用跨数据集表：" + md[s:mislabel.end() + 10].replace("\n", " "))
    # 2) 跨数据集对比表「素材未提供」的笼统说法（上下文可能确为单数据集，但措辞易误导）
    for m in re.finditer(r"素材未提供跨数据集对比表", md):
        s = max(0, m.start() - 20)
        flags.append("跨数据集表缺失声明：" + md[s:m.end() + 10].replace("\n", " "))
    # 3) 在有真实数字处却声称「素材未提供 … 无法 …」的强谎言（须同时出现未提供+无法）
    for m in re.finditer(r"素材未提供.{0,40}无法", md):
        s = max(0, m.start() - 20)
        flags.append("未提供且无法：" + md[s:m.end() + 10].replace("\n", " "))
    return flags


def main():
    if len(sys.argv) < 2:
        print("用法: python audit_consistency.py <paper_full.md>")
        return 2
    path = Path(sys.argv[1])
    md = path.read_text(encoding="utf-8")
    tbl = parse_verdict_table(md)
    concl = parse_conclusion(md)
    print(f"[审计] {path}")
    print(f"  裁决表提取: {tbl}")
    print(f"  结论提取  : {concl}")

    problems = []
    all_h = sorted(set(tbl) | set(concl), key=lambda k: int(re.sub(r'\D', '', k) or 0))
    for h in all_h:
        tv = (tbl.get(h) or "").strip()
        cv = (concl.get(h) or "").strip()
        if not tv or not cv:
            continue
        # 归一化：表可能是「证据不足」，结论可能是「证据不足」
        if tv != cv:
            problems.append(f"  ✗ {h} 不一致：裁决表='{tv}' vs 结论='{cv}'")

    lies = detect_lies(md)
    if lies:
        for l in lies:
            problems.append(f"  ✗ 疑似素材谎言: …{l}…")

    if problems:
        print("\n[结果] 发现 {n} 处问题：".format(n=len(problems)))
        for p in problems:
            print(p)
        return 1
    print("\n[结果] ✅ 通过：裁决表与结论完全一致，未发现素材谎言。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
