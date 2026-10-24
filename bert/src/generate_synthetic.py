#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 DeepSeek 生成 1 级(垃圾/违规)虚拟弹幕, 补足训练集样本量。

- 以现有真实 1 级样本为 few-shot 种子, 让 LLM 生成同风格弹幕
- 广告引流 / 辱骂人身攻击 / 低俗色情 / 政治敏感 四类各生成一批
- 去重(生成内部 + 与现有标注数据)

用法: python bert/src/generate_synthetic.py [--target 1500 --batch 20]
输出: bert/raw_data/synthetic_level1.jsonl  {"text": ..., "label": 1(语义)}
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

BERT_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BERT_DIR / "raw_data" / "synthetic_level1.jsonl"
EXCLUDE_FILES = [BERT_DIR / "raw_data" / "train_data.jsonl",
                 BERT_DIR / "raw_data" / "val_data.jsonl",
                 BERT_DIR / "raw_data" / "test_data.jsonl"]

# 真实 1 级种子样本(从现有标注数据中抽取)
SEED_EXAMPLES = [
    "加V看片:xxxxx", "视频同款立即购买", "导演是鲨比吗?", "你他妈放好的啊",
    "狗东西一个", "傻逼玩意儿", "刷你妈CP", "活该啊！一群叛徒就该当街乱棍打死！",
    "拼多多，淘宝，回复客服一小时30~35，有想试试的吗", "上拼多多",
    "脏死了下面不得长成花菜林", "我想嫖娼", "四全家穷一辈子接不接",
    "沙比", "一帮狗东西", "这个视频发反弹的全家替我挡灾", "戴上看片",
    "已下单（玩大哥担保）", "再问有没有人扫码", "id：飞利浦球球，你这种地域黑全家都是唐氏症",
]

# 四类生成指令
GEN_TYPES = [
    ("广告引流", "广告、引流、卖片、联系方式、推广话术"),
    ("辱骂攻击", "辱骂、人身攻击、诅咒、贬低他人"),
    ("低俗色情", "低俗、色情、性暗示、下流话"),
    ("政治敏感", "政治敏感、恶意引战、违规言论"),
]

SYSTEM_PROMPT = """你是弹幕生成器。根据给定的风格和示例, 生成弹幕平台上的【垃圾/违规弹幕】。
要求:
1. 每条 1~30 个字, 口语化, 像真人发的弹幕, 不要重复示例原文
2. 风格多样: 有的直白, 有的隐晦, 有的玩梗式攻击, 有的像小广告
3. 只输出 JSON 数组, 每项是字符串, 不要输出其他内容"""


def load_existing_texts():
    texts = set()
    for p in EXCLUDE_FILES:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    texts.add(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return texts


def build_user_prompt(gen_name, gen_desc, n):
    seeds = "\n".join(f"- {s}" for s in SEED_EXAMPLES[:12])
    return (f"请生成 {n} 条【{gen_name}】类违规弹幕({gen_desc})。\n"
            f"风格参考(不要照抄):\n{seeds}\n"
            f"生成要求: 每条独立不重复, 返回 JSON 数组。")


def parse_reply(text):
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("回复里没有 JSON 数组")
    arr = json.loads(t[start:end + 1])
    return [str(x).strip() for x in arr]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="生成 1 级虚拟弹幕")
    parser.add_argument("--target", type=int, default=1500, help="目标生成条数")
    parser.add_argument("--batch", type=int, default=20, help="每批生成条数")
    args = parser.parse_args()

    # API 配置(复用 settings.json 的 DeepSeek)
    settings_path = Path.home() / ".claude" / "settings.json"
    env = json.loads(settings_path.read_text(encoding="utf-8")).get("env", {})
    client = OpenAI(api_key=env["ANTHROPIC_AUTH_TOKEN"],
                    base_url="https://api.deepseek.com")
    model = "deepseek-chat"

    existing = load_existing_texts()
    print(f"排除已存在文本 {len(existing)} 条")

    # 断点续生成
    generated = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            try:
                generated.add(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"已有生成 {len(generated)} 条, 继续补充")

    needed = args.target - len(generated)
    round_idx = 0
    while needed > 0 and round_idx < 30:   # 上限 30 轮, 防死循环
        round_idx += 1
        got_round = 0
        for gen_name, gen_desc in GEN_TYPES:
            if needed <= 0:
                break
            n_this = max(1, min(args.batch, needed))
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(gen_name, gen_desc, n_this)},
                    ],
                    temperature=1.0,
                    max_tokens=1200,
                )
                items = parse_reply(resp.choices[0].message.content)
            except Exception as e:
                print(f"  [{gen_name}] 生成失败: {str(e)[:80]}")
                time.sleep(2)
                continue
            fresh = 0
            with open(OUT_PATH, "a", encoding="utf-8") as f:
                for t in items:
                    t = t.strip().strip('"').strip("，。")
                    if not t or len(t) > 30 or len(t) < 2:
                        continue
                    if t in existing or t in generated:
                        continue
                    generated.add(t)
                    fresh += 1
                    f.write(json.dumps({"text": t, "label": 1}, ensure_ascii=False) + "\n")
            got_round += fresh
            print(f"  [{gen_name}] 本批新增 {fresh} 条")
            time.sleep(0.5)
        needed = args.target - len(generated)
        print(f"第 {round_idx} 轮完成, 累计 {len(generated)}/{args.target}, 仍需 {needed}")

    print(f"\n完成: 共生成 {len(generated)} 条 1 级虚拟弹幕 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
