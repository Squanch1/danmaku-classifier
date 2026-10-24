#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1/2 级弹幕定向挖掘(训练集补采)。

- 1 级: 广告引流 / 辱骂攻击 关键词命中(保守词组, 降低误挖)
- 2 级: 灌水模式(纯"哈哈"、纯表情、打卡、前排、+1、纯数字/字母刷屏)
- 排除全部已标注文本

用法: python data/scripts/mine_low_quality.py [--n1 700 --n2 1500]
输出: data/candidates_low.jsonl(带 hit_type: "l1"/"l2")
"""

import argparse
import json
import re
import sys
from pathlib import Path

from config import Config

conf = Config()
DATA_DIR = Path(__file__).resolve().parent.parent
RAW = DATA_DIR / "danmaku_raw.jsonl"
OUT = DATA_DIR / "candidates_low.jsonl"

EXCLUDE = [DATA_DIR / "pilot_labeled.jsonl",
           DATA_DIR / "candidates_high_labeled.jsonl",
           DATA_DIR / "test_natural.jsonl",
           DATA_DIR / "val_natural.jsonl"]

# 1 级关键词(广告引流)—— 第二轮: 从 46 条已确认样本提炼, 去掉了"货源/主页"等误匹配词
AD_PATTERNS = [r"加\s*[vV微信]", r"微信\s*\d", r"Q{2}群", r"qq群", r"淘宝", r"拼多多",
               r"代购", r"代练", r"私信", r"私聊", r"点我头像", r"打钱", r"充值",
               r"看片", r"卖片", r"种\s*子", r"扫码", r"下单", r"客服", r"刷单",
               r"代充", r"v信", r"薇信", r"VX", r"简介里", r"链接在简介", r"回复.*小时",
               r"li?nks?\b", r"ADS", r"zi sao"]
# 1 级关键词(辱骂攻击)
ABUSE_PATTERNS = [r"傻\s*逼", r"煞笔", r"沙比", r"你\s*他\s*妈", r"你\s*吗", r"你妈",
                  r"草泥马", r"去死", r"畜生", r"狗东西", r"狗比", r"滚出", r"滚蛋",
                  r"死全家", r"四全家", r"孤儿", r"舔狗", r"贱人", r"婊子", r"贱货",
                  r"嫖", r"招嫖", r"卖淫", r"卖身", r"活该", r"白痴", r"脑残", r"智障",
                  r"唐氏", r"吃屎", r"大便", r"泼妇", r"傻叉", r"猪头肉", r"垃圾人"]
# 2 级灌水模式
WATER_PATTERNS = [r"^哈{2,}$", r"^[哈嘿嘻呵]{2,}$", r"^\+1\+?$", r"^\+1$", r"^打卡$", r"^前排$",
                  r"^签到$", r"^路过$", r"^6{2,}$", r"^[a-zA-Z]{2,}$",
                  r"^[0-9]{2,}$", r"^[^一-鿿]{2,12}$", r"^欧了$", r"^沙发$",
                  r"^火钳刘明$", r"^[哦嗯啊噢咦嚯]{2,}$"]

AD_RE = [re.compile(p) for p in AD_PATTERNS]
ABUSE_RE = [re.compile(p) for p in ABUSE_PATTERNS]
WATER_RE = [re.compile(p) for p in WATER_PATTERNS]


def classify(text):
    """返回 (hit_type, 命中词列表)。优先 1 级。"""
    if any(r.search(text) for r in AD_RE):
        return "l1", "广告"
    if any(r.search(text) for r in ABUSE_RE):
        return "l1", "辱骂"
    if any(r.search(text) for r in WATER_RE):
        return "l2", "灌水"
    return None, None


def load_excluded():
    texts = set()
    for p in EXCLUDE:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    texts.add(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return texts


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1", type=int, default=700, help="1 级候选目标数")
    parser.add_argument("--n2", type=int, default=1500, help="2 级候选目标数")
    args = parser.parse_args()

    excluded = load_excluded()
    print(f"排除已标注文本 {len(excluded)} 条")

    l1, l2 = [], []
    seen = set()
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row["text"]
        if text in excluded or text in seen:
            continue
        seen.add(text)
        hit_type, reason = classify(text)
        if hit_type == "l1" and len(l1) < args.n1:
            row["hit_type"] = "l1"
            row["hit_reason"] = reason
            l1.append(row)
        elif hit_type == "l2" and len(l2) < args.n2:
            row["hit_type"] = "l2"
            row["hit_reason"] = reason
            l2.append(row)
        if len(l1) >= args.n1 and len(l2) >= args.n2:
            break

    cands = l1 + l2
    OUT.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cands),
                   encoding="utf-8")
    print(f"1 级候选: {len(l1)} | 2 级候选: {len(l2)} | 合计 {len(cands)} -> {OUT}")
    for c in l1[:5]:
        print(f"  [1级/{c['hit_reason']}] {c['text'][:36]}")
    for c in l2[:5]:
        print(f"  [2级/{c['hit_reason']}] {c['text'][:36]}")


if __name__ == "__main__":
    main()
