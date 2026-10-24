#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从原始池抽取自然分布验证集/测试集样本(排除全部已标注文本)。

- 测试集 5000 条 + 验证集 1000 条, 纯随机抽样(自然分布)
- 排除: pilot_labeled / candidates_high_labeled / quality_labeled 全部已标文本
- 固定种子, 可复现

用法: python data/scripts/sample_natural.py
输出: data/test_natural_sample.jsonl(5000) + data/val_natural_sample.jsonl(1000)
"""

import csv
import json
import random
import sys
from pathlib import Path

from config import Config

conf = Config()
DATA_DIR = Path(__file__).resolve().parent.parent
RAW = DATA_DIR / "danmaku_raw.jsonl"
OUT_TEST = DATA_DIR / "test_natural_sample.jsonl"
OUT_VAL = DATA_DIR / "val_natural_sample.jsonl"
TEST_N = 5000
VAL_N = 1000
SEED = conf.seed

EXCLUDE = [DATA_DIR / "pilot_labeled.jsonl",
           DATA_DIR / "candidates_high_labeled.jsonl"]


def load_excluded():
    texts = set()
    for p in EXCLUDE:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    texts.add(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    pass
    for p in sorted((DATA_DIR / "scripts").glob("quality_labeled_*.csv")):
        with open(p, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("内容"):
                    texts.add(row["内容"])
    return texts


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    excluded = load_excluded()
    print(f"排除已标注文本 {len(excluded)} 条")

    rows = []
    seen = set()
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["text"] in excluded or row["text"] in seen:
            continue
        seen.add(row["text"])
        rows.append(row)
    print(f"原始池去重且未标注: {len(rows)} 条")

    rng = random.Random(SEED)
    rng.shuffle(rows)
    test_part = rows[:TEST_N]
    val_part = rows[TEST_N:TEST_N + VAL_N]

    for path, part, name in [(OUT_TEST, test_part, "测试集"),
                             (OUT_VAL, val_part, "验证集")]:
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in part),
            encoding="utf-8")
        # 长度/mode 快照
        lens = [len(r["text"]) for r in part]
        modes = {}
        for r in part:
            modes[r.get("mode")] = modes.get(r.get("mode"), 0) + 1
        print(f"{name}: {len(part)} 条 -> {path} | 平均长度 {sum(lens)/len(lens):.1f} | mode {modes}")


if __name__ == "__main__":
    main()
