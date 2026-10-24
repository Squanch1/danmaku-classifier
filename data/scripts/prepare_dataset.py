#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并已标注数据并分层划分 train/val/test。

输入(自动去重, 先出现的优先):
  1. pilot_labeled.jsonl            —— 分层抽样的试标注
  2. candidates_high_labeled.jsonl  —— 定向补采候选池标注
  3. scripts/quality_labeled_*.csv  —— label_llm.py 批量标注汇总
输出: train_data.jsonl(合并去重) + train/val/train.jsonl(按标签分层, 70/15/15)

用法: python data/scripts/prepare_dataset.py
"""

import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from config import Config

conf = Config()

DATA_DIR = Path(__file__).resolve().parent.parent   # 脚本在 data/scripts/, 数据在 data/
JSONL_INPUTS = [conf.pilot_labeled_path, DATA_DIR / "candidates_high_labeled.jsonl"]
CSV_PATTERN = str(DATA_DIR / "scripts" / "quality_labeled_*.csv")
OUT_MERGED = DATA_DIR / "train_data.jsonl"
SPLIT_PATHS = {
    "train": (DATA_DIR / "test.jsonl", 0.70),
    "val": (DATA_DIR / "val.jsonl", 0.15),
    "test": (DATA_DIR / "train.jsonl", 0.15),
}
SEED = conf.seed


def norm_label(label):
    if label == 5:
        label = 4          # 旧方案 5 级并入 4 级
    return label if label in (1, 2, 3, 4) else None


def load_jsonl(file_path):
    rows = {}
    for line in Path(file_path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        label = norm_label(row.get("label"))
        if label:
            rows[row["text"]] = {"text": row["text"], "label": label}
    return rows


def load_quality_csv(file_path):
    rows = {}
    with open(file_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                label = norm_label(int(row["标签"]))
            except (KeyError, ValueError):
                continue
            if label and row.get("内容"):
                rows[row["内容"]] = {"text": row["内容"], "label": label}
    return rows


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    merged = {}
    for f in JSONL_INPUTS:
        if not Path(f).exists():
            print(f"[warn] 跳过不存在的文件: {f}")
            continue
        rows = load_jsonl(f)
        print(f"{f}: 有效标注 {len(rows)} 条")
        merged.update(rows)   # 相同文本以先出现的文件为准

    for f in sorted(glob.glob(CSV_PATTERN)):
        rows = load_quality_csv(f)
        print(f"{f}: 有效标注 {len(rows)} 条")
        merged.update(rows)

    items = list(merged.values())
    dist = Counter(r["label"] for r in items)
    print(f"\n合并去重后: {len(items)} 条 | 分布: {{ {', '.join(f'{k}:{v}' for k, v in sorted(dist.items()))} }}")

    OUT_MERGED.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in items), encoding="utf-8")

    # 分层划分: 先切出 train,再对剩余 50/50 切 val/test
    labels = [r["label"] for r in items]
    train, rest = train_test_split(
        items, test_size=1 - SPLIT_PATHS["train"][1],
        stratify=labels, random_state=SEED)
    rest_labels = [r["label"] for r in rest]
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest_labels, random_state=SEED)

    for key, part in [("train", train), ("val", val), ("test", test)]:
        path, _ = SPLIT_PATHS[key]
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in part), encoding="utf-8")
        d = Counter(r["label"] for r in part)
        print(f"{path}: {len(part)} 条 | 分布: {{ {', '.join(f'{k}:{v}' for k, v in sorted(d.items()))} }}")


if __name__ == "__main__":
    main()
