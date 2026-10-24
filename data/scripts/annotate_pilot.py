#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pilot 样本 LLM 标注脚本(仅标准库)

读取 pilot_sample.jsonl,按五级质量规范逐条调用 LLM(Anthropic 兼容接口,
默认复用 Claude Code settings.json 里配置的 DeepSeek)打标,输出
pilot_labeled.jsonl。支持断点续跑(已成功标注的不重复标)与人工抽检导出。

用法:
    python annotate_pilot.py --limit 5     # 冒烟测试
    python annotate_pilot.py               # 全量标注(默认 4 并发)
    python annotate_pilot.py --review 100  # 标注后导出 100 条供人工抽检
    python annotate_pilot.py --workers 1   # 单线程(最稳)
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import Config

conf = Config()

# ---------------------------------------------------------------------------
# 标注规范(五级, 数字越高质量越好)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是弹幕质量标注专家。对用户给出的每条弹幕标注质量等级,输出 JSON: {"label": <1-4的整数>, "reason": "<一句话理由,不超过30字>"}。

等级定义:
- 1级(垃圾/违规): 广告引流、卖片、联系方式/网址、辱骂人身攻击、低俗色情、政治敏感
- 2级(低质): 无信息量的灌水,如纯"哈哈哈哈"、纯表情颜文字、"打卡"、"+1"、无意义数字串、与视频内容完全无关的水评
- 3级(普通): 正常互动与情绪表达,如"666"、"前方高能"、剧情/画面吐槽、队形参与、简单提问或回应、情感共鸣
- 4级(优质/精华): 有信息量到高价值,包括剧情细节补充、有观点的相关讨论、完整句子的外文翻译、科普解释与纠错、BGM/出处提示、关键时间点导航(带时间戳)、独到见解或创造梗

边界裁决规则:
1. 纯"哈哈哈哈"/纯表情,无论多长 = 2级
2. "哈哈哈"+有内容的吐槽 = 3级
3. 简短常见互动("666"、"前方高能"、"名场面"、"泪目") = 3级
4. 完整句子以上的翻译 = 4级,不再区分句子数量
5. 指出视频中错误并给出正确信息 = 4级
6. 广告/联系方式/网址 = 1级,无例外
7. 与视频无关的灌水 = 2级;表达对视频/up主的情绪 = 3级
8. 拿不准时取保守值 3 级

Few-shot 示例:
"哈哈哈哈哈哈" -> {"label": 2, "reason": "纯情绪无信息量"}
"前方高能预警" -> {"label": 3, "reason": "常见互动提示"}
"66666" -> {"label": 3, "reason": "常见赞赏互动"}
"加V看片:xxxxx" -> {"label": 1, "reason": "广告引流"}
"这里翻译:他说的是'我们会永远在一起'" -> {"label": 4, "reason": "完整一句翻译"}
"科普:视频里把第二定律讲错了,正确的表述是……" -> {"label": 4, "reason": "纠错科普"}
"up主说的对,这里其实是致敬了XX电影" -> {"label": 4, "reason": "补充相关知识"}
"打卡" -> {"label": 2, "reason": "无信息量灌水"}
"淦,这也太帅了吧" -> {"label": 3, "reason": "情绪表达"}

只输出 JSON,不要输出其他内容。"""


# ---------------------------------------------------------------------------
# API 配置与调用
# ---------------------------------------------------------------------------

def load_api_config():
    """按优先级取 API 配置: 命令行参数 > 进程环境变量 > ~/.claude/settings.json。"""
    cfg = {
        "base_url": os.environ.get("ANTHROPIC_BASE_URL", ""),
        "token": os.environ.get("ANTHROPIC_AUTH_TOKEN", ""),
        "model": os.environ.get("ANTHROPIC_MODEL", ""),
    }
    if not (cfg["base_url"] and cfg["token"]):
        settings_path = Path.home() / ".claude" / "settings.json"
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            env = settings.get("env", {})
            cfg = {
                "base_url": env.get("ANTHROPIC_BASE_URL", cfg["base_url"]),
                "token": env.get("ANTHROPIC_AUTH_TOKEN", cfg["token"]),
                "model": env.get("ANTHROPIC_MODEL", cfg["model"]),
            }
        except (OSError, json.JSONDecodeError):
            pass
    return cfg


def call_llm(text, title, cfg, temperature=0.0):
    """调用 Anthropic 兼容接口(Messages API),返回模型回复文本。"""
    url = cfg["base_url"].rstrip("/") + "/v1/messages"
    payload = {
        "model": cfg["model"],
        "max_tokens": 300,
        "temperature": temperature,
        "thinking": {"type": "disabled"},   # DeepSeek 默认带思考块, 关闭以省 token
        "system": SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": f"视频:《{title}》\n弹幕: {text}\n请标注这条弹幕的等级。",
        }],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": cfg["token"],
            "Authorization": f"Bearer {cfg['token']}",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    for block in body.get("content", []):   # 兼容 thinking/text 块混排的返回
        if block.get("type") == "text":
            return block["text"]
    raise KeyError("响应中没有 text 块")


def parse_label(reply):
    """从模型回复中解析 {"label": n, "reason": "..."},失败返回 None。"""
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    label = obj.get("label")
    if isinstance(label, str):
        digit = re.search(r"[1-4]", label)
        label = int(digit.group(0)) if digit else None
    if label not in (1, 2, 3, 4):
        return None
    reason = str(obj.get("reason", ""))[:200]
    return label, reason


def label_one(text, title, cfg, temperature):
    """单条标注,3 次重试;最终失败抛出异常。"""
    last_err = None
    for attempt in range(3):
        try:
            reply = call_llm(text, title, cfg, temperature)
            parsed = parse_label(reply)
            if parsed:
                return parsed
            last_err = RuntimeError(f"回复无法解析: {reply[:60]}")
        except (urllib.error.URLError, OSError, KeyError, IndexError, json.JSONDecodeError) as e:
            last_err = e
        time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(str(last_err))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def load_annotated(path):
    """读取已有标注结果,返回 {text: (label, reason)} 及全部有效行。"""
    done, valid_rows = {}, []
    if not Path(path).exists():
        return done, valid_rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        label = row.get("label")
        if label == 5:
            label = 4                       # 旧方案 5 级并入 4 级
            row["label"] = 4
        if label in (1, 2, 3, 4):
            done[row["text"]] = (label, row.get("label_reason", ""))
            valid_rows.append(row)
    return done, valid_rows


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="pilot 样本 LLM 标注")
    parser.add_argument("--limit", type=int, default=0, help="只标前 N 条(0=全部)")
    parser.add_argument("--workers", type=int, default=4, help="并发数")
    parser.add_argument("--review", type=int, default=0, help="标注后导出 N 条人工抽检样本(0=不导出)")
    parser.add_argument("--review-out", default=conf.review_sample_path, help="抽检样本输出路径")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", default="", help="API key(默认取环境变量/settings.json)")
    parser.add_argument("--base-url", default="", help="API 地址(默认取环境变量/settings.json)")
    parser.add_argument("--model", default="", help="模型名(默认取环境变量/settings.json)")
    parser.add_argument("--input", default=conf.pilot_sample_path)
    parser.add_argument("--out", default=conf.pilot_labeled_path)
    args = parser.parse_args()

    cfg = load_api_config()
    if args.api_key:
        cfg["token"] = args.api_key
    if args.base_url:
        cfg["base_url"] = args.base_url
    if args.model:
        cfg["model"] = args.model
    if not (cfg["base_url"] and cfg["token"] and cfg["model"]):
        sys.exit("未找到 API 配置。请设置环境变量 ANTHROPIC_BASE_URL/ANTHROPIC_AUTH_TOKEN/ANTHROPIC_MODEL,"
                 "或通过 --api-key/--base-url/--model 传入。")
    print(f"API: {cfg['base_url']} | 模型: {cfg['model']} | 并发: {args.workers}")

    samples = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    done, valid_rows = load_annotated(args.out)
    todo = [s for s in samples if s["text"] not in done]
    if args.limit:
        todo = todo[:max(0, args.limit - len(done))]
    print(f"输入 {len(samples)} 条 | 已标注 {len(done)} 条 | 待标注 {len(todo)} 条")

    out_path = Path(args.out)
    write_lock = threading.Lock()
    progress_lock = threading.Lock()
    counters = {"done": 0, "fail": 0}

    def safe_append(row):
        with write_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def process(sample):
        text, title = sample["text"], sample.get("title", "")
        try:
            label, reason = label_one(text, title, cfg, args.temperature)
            row = dict(sample)
            row["label"] = label
            row["label_reason"] = reason
            safe_append(row)
            with progress_lock:
                counters["done"] += 1
        except Exception as e:                       # 标记为 -1,下次运行自动重标
            row = dict(sample)
            row["label"] = -1
            row["label_reason"] = str(e)[:100]
            safe_append(row)
            with progress_lock:
                counters["fail"] += 1
        with progress_lock:
            done_now = counters["done"]
            if done_now % 20 == 0 or done_now == len(todo):
                print(f"  进度: 成功 {done_now}/{len(todo)} | 失败 {counters['fail']}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, s) for s in todo]
        for _ in as_completed(futures):
            pass

    elapsed = time.time() - t0
    print(f"\n耗时 {elapsed:.0f}s | 成功 {counters['done']} | 失败 {counters['fail']}"
          f"(失败条目 label=-1,重跑本脚本会自动重标)")

    # 标注后的小规模有标签 EDA: 等级分布
    done, rows = load_annotated(args.out)
    dist = {}
    for r in rows:
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    print("等级分布:", {k: dist.get(k, 0) for k in range(1, 5)})

    if args.review:
        random.seed(conf.seed)
        review = random.sample(rows, min(args.review, len(rows)))
        Path(args.review_out).write_text(
            "\n".join(json.dumps({
                "text": r["text"], "label": r["label"],
                "label_reason": r.get("label_reason", ""),
                "mode": r.get("mode"), "bvid": r.get("bvid"),
                "title": r.get("title"),
            }, ensure_ascii=False) for r in review),
            encoding="utf-8")
        print(f"抽检样本已导出 {len(review)} 条 -> {args.review_out}")


if __name__ == "__main__":
    main()
