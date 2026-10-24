#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高质量弹幕候选池定向挖掘(方案A: 4~5 级定向补采)

从 20 万原始池按三类信号挖掘 4~5 级候选:
  1) mode=4 底部弹幕 —— 字幕/翻译/注释高产区, 总体仅 0.4%
  2) 关键词模式 —— 翻译/科普纠错/时间戳导航/出处提示
  3) 长文本(>=30 字)

输出 candidates_high.jsonl(按得分降序), 附 hits(命中的规则)与 score,
可直接交给 annotate_pilot.py 标注。已标注过的 pilot 文本自动排除。

用法:
    python mine_candidates.py                  # 全量挖掘
    python mine_candidates.py --top 3000       # 只保留得分最高的 3000 条
    python mine_candidates.py --len 20         # 调长文本阈值
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from config import Config

conf = Config()

# 每类信号一个命名正则;命中即得分
SIGNAL_PATTERNS = {
    "翻译": re.compile(r"翻译|字幕|译注|注[:：]|这句|这句(话)?(的)?意思|他说|TA说|歌词[:：]"),
    "科普纠错": re.compile(r"科普|其实|事实上|正确的是|应该是|注意|纠正|错了|不对|辟谣|冷知识|原理|依据"),
    "时间戳导航": re.compile(r"(?<!\d)\d{1,2}:\d{2}(?!\d)"),   # 如 12:34
    "出处提示": re.compile(r"出自|出处|来源|BGM|bgm|原曲|插曲|片头曲|片尾曲|这首歌|背景音乐"),
}

MODE_SCORE = 2      # 底部弹幕加权(信号最强)
LEN_SCORE = 1       # 长文本加权


def hit_signals(text, length, mode, min_len):
    hits, score = [], 0
    if mode == 4:
        hits.append("mode4")
        score += MODE_SCORE
    for name, pat in SIGNAL_PATTERNS.items():
        if pat.search(text):
            hits.append(name)
            score += 1
    if length >= min_len:
        hits.append(f"len>={min_len}")
        score += LEN_SCORE
    return hits, score


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="高质量弹幕候选挖掘")
    parser.add_argument("--input", default=conf.raw_data_path)
    parser.add_argument("--out", default=Path(__file__).resolve().parent.parent / "candidates_high.jsonl")
    parser.add_argument("--len", dest="min_len", type=int, default=30, help="长文本阈值(字)")
    parser.add_argument("--top", type=int, default=0, help="只保留得分最高的 N 条(0=全部)")
    parser.add_argument("--exclude", default=conf.pilot_labeled_path, help="排除已标注文件")
    args = parser.parse_args()

    # 排除 pilot 已标注的文本,避免重复标注
    labeled_texts = set()
    exclude_path = Path(args.exclude)
    if exclude_path.exists():
        for line in exclude_path.read_text(encoding="utf-8").splitlines():
            try:
                labeled_texts.add(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"排除已标注文本 {len(labeled_texts)} 条")

    candidates = []
    rule_counter = Counter()
    seen = set()
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text, mode = row["text"], row.get("mode", 0)
        if text in seen or text in labeled_texts:
            continue
        seen.add(text)
        hits, score = hit_signals(text, len(text), mode, args.min_len)
        if score <= 0:
            continue
        for h in hits:
            rule_counter[h] += 1
        candidates.append({**row, "hits": hits, "score": score})

    candidates.sort(key=lambda r: (-r["score"], -len(r["text"])))
    if args.top:
        candidates = candidates[:args.top]

    Path(args.out).write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in candidates),
        encoding="utf-8")
    print(f"候选池: {len(candidates)} 条 -> {args.out}")
    print("各规则命中数(允许重叠):")
    for rule, cnt in rule_counter.most_common():
        print(f"  {rule:<12} {cnt:>7,}")
    if candidates:
        print(f"得分分布: {Counter(c['score'] for c in candidates)}")
        print(f"候选长度: 最短 {min(len(c['text']) for c in candidates)} | "
              f"平均 {sum(len(c['text']) for c in candidates) / len(candidates):.0f}")
        print("\n高分样例(score>=4, 前 10):")
        n = 0
        for c in candidates:
            if c["score"] >= 4:
                print(f"  [{c['score']} {','.join(c['hits'])}] {c['text'][:50]}")
                n += 1
                if n >= 10:
                    break


if __name__ == "__main__":
    main()
