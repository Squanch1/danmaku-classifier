#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新划分数据集(方案 B): 测试/验证 1:1:1:1 均衡, 1级混入虚拟弹幕。

数据源:
  raw_data/train_data.jsonl        真实标注(语义标签 1~4)
  raw_data/synthetic_level1.jsonl  虚拟 1 级(语义标签 1)

划分:
  测试集: 每类 1000 条 = 4000
  验证集: 每类 250 条  = 1000
  训练集: 剩余全部

输出:
  raw_data/{test_data,val_data,train_data}.jsonl      语义标签 1~4
  processed_data/{test,val,train}.jsonl               0~3 标签(供 train.py 读取)

用法: python bert/src/resplit_dataset.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

BERT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BERT_DIR / "raw_data"
PROC_DIR = BERT_DIR / "processed_data"

TEST_PER_CLASS = 1000
VAL_PER_CLASS = 250
TRAIN_PER_CLASS = 400   # 训练集也同分布 1:1:1:1
SEED = 42


def load_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    # 1.合并数据
    real = load_jsonl(RAW_DIR / "train_data.jsonl")
    synthetic = load_jsonl(RAW_DIR / "synthetic_level1.jsonl")
    print(f"真实标注 {len(real)} 条 | 虚拟 1 级 {len(synthetic)} 条")

    merged = {}
    for r in real + synthetic:
        text, label = r["text"], int(r["label"])
        if label not in (1, 2, 3, 4):
            continue
        merged[text] = label     # text 去重(虚拟已排除现有文本, 此步兜底)
    items = [{"text": t, "label": l} for t, l in merged.items()]
    print(f"合并去重后 {len(items)} 条 | 分布: {dict(Counter(l for _, l in merged.items()))}")

    # 2.按标签分组, 组内洗牌
    rng = random.Random(SEED)
    by_label = {lab: [] for lab in (1, 2, 3, 4)}
    for it in items:
        by_label[it["label"]].append(it)
    for lab in by_label:
        rng.shuffle(by_label[lab])
        print(f"  {lab}级可用: {len(by_label[lab])} 条")

    # 3.切分(测试 1000 + 验证 250 + 训练 400, 每类同分布)
    test, val, train = [], [], []
    for lab in (1, 2, 3, 4):
        pool = by_label[lab]
        test += pool[:TEST_PER_CLASS]
        val += pool[TEST_PER_CLASS:TEST_PER_CLASS + VAL_PER_CLASS]
        train += pool[TEST_PER_CLASS + VAL_PER_CLASS:TEST_PER_CLASS + VAL_PER_CLASS + TRAIN_PER_CLASS]
        leftover = len(pool) - TEST_PER_CLASS - VAL_PER_CLASS - TRAIN_PER_CLASS
        if leftover > 0:
            print(f"  [提示] {lab}级剩余 {leftover} 条未使用(如需要可改大 TRAIN_PER_CLASS)")

    # 4.输出(raw_data 语义 1~4; processed_data 0~3 供训练)
    for name, part in [("test_data", test), ("val_data", val), ("train_data", train)]:
        path = RAW_DIR / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in part), encoding="utf-8")
        d = Counter(r["label"] for r in part)
        print(f"{path}: {len(part)} 条 | {dict(sorted(d.items()))}")

    short_name = {"test_data": "test", "val_data": "val", "train_data": "train"}
    PROC_DIR.mkdir(exist_ok=True)
    for name, part in [("test_data", test), ("val_data", val), ("train_data", train)]:
        path = PROC_DIR / f"{short_name[name]}.jsonl"
        path.write_text("\n".join(
            json.dumps({"text": r["text"], "label": r["label"] - 1}, ensure_ascii=False)
            for r in part), encoding="utf-8")
        d = Counter(r["label"] - 1 for r in part)
        print(f"{path}: {len(part)} 条(0~3) | {dict(sorted(d.items()))}")

    print(f"\n完成: 测试 {len(test)} | 验证 {len(val)} | 训练 {len(train)}")


if __name__ == "__main__":
    main()
