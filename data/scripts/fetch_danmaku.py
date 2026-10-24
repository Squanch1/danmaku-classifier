#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B站弹幕采集脚本(仅依赖 Python 标准库)

流程: 热门视频列表 -> 获取视频 cid -> 下载弹幕池 XML(明文或 raw-deflate 压缩)
     -> 解码(UTF-8 / GBK 启发式) -> 去重 -> 输出 JSONL

用法示例:
    python fetch_danmaku.py                     # 默认采集 20000 条到 danmaku_raw.jsonl
    python fetch_danmaku.py --target 5000 --pages 2
    python fetch_danmaku.py --target 1000 --out train.jsonl --max-videos 5

说明:
    - 数据来自 B站公开弹幕接口(comment.bilibili.com),仅建议用于学习/研究用途,
      控制请求频率,勿用于商业用途。
"""

import argparse
import html
import json
import random
import re
import sys
import time
import urllib.request
import urllib.error
import zlib
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量与 HTTP 基础
# ---------------------------------------------------------------------------

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Referer": "https://www.bilibili.com/",
    "Accept": "*/*",
}
TIMEOUT = 30

# 弹幕 XML 里 <d p="..."> 的 p 字段顺序:
# progress,mode,fontsize,color,ctime,pool,user_hash,row_id
D_PATTERN = re.compile(rb'<d p="([^"]*)">(.*?)</d>', re.S)


def http_get(url, retries=3, timeout=TIMEOUT):
    """带重试和退避的 GET,返回 bytes。"""
    last_err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError) as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET {url} 失败: {last_err}")


def http_get_json(url):
    return json.loads(http_get(url).decode("utf-8", "replace"))


# ---------------------------------------------------------------------------
# 弹幕文本编码处理
# ---------------------------------------------------------------------------

def _cjk_ratio(s: str) -> float:
    """CJK 统一表意文字占比。"""
    if not s:
        return 0.0
    return sum(1 for c in s if "一" <= c <= "鿿") / len(s)


def _looks_like_gbk_as_utf8(s: str) -> bool:
    """GBK 中文被误读为 UTF-8 的特征。

    GBK 字节对恰好是合法 UTF-8 序列时,解码结果落在拉丁扩展 / 西里尔 /
    希腊 / 希伯来 / 阿拉伯 / 泰文等"外来文字区",正常中文弹幕不会出现
    这些字符(日文假名、emoji 不受影响,不会被误判)。
    """
    for c in s:
        o = ord(c)
        if 0x00C0 <= o <= 0x02AF or 0x0370 <= o <= 0x052F or 0x0590 <= o <= 0x07FF:
            return True
    return False


def decode_danmaku_text(raw: bytes) -> str:
    """弹幕文本解码。

    B站老弹幕池存在两个编码坑:
    1. 部分弹幕是 GBK 编码但 XML 声明 UTF-8,直接 UTF-8 解码报错;
    2. GBK 字节序列常"恰好"是合法 UTF-8,解码不报错但结果是乱码
       (外来文字区字符),而 UTF-8 中文被误读为 GBK 则得到生僻汉字
       (如 缁欐垜鐪嬫垚)。
    策略: 优先严格 UTF-8;若结果出现外来文字区字符且 GBK 解出的中文
    占比足够高,则改用 GBK。反之(正常中文/英文/日文/emoji)保持 UTF-8。
    """
    try:
        utf8_text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gb18030", "replace")

    if _looks_like_gbk_as_utf8(utf8_text):
        try:
            gbk_text = raw.decode("gb18030")
        except UnicodeDecodeError:
            return utf8_text
        # 阈值防止把"真外语 + 个别重音字母"的正常弹幕误判为 GBK
        if _cjk_ratio(gbk_text) >= 0.5:
            return gbk_text
    return utf8_text


# ---------------------------------------------------------------------------
# B站接口
# ---------------------------------------------------------------------------

def get_popular_videos(pages=3, ps=50):
    """热门视频列表,按弹幕数降序(优先采集弹幕多的视频,减少请求量)。"""
    videos = []
    for pn in range(1, pages + 1):
        try:
            data = http_get_json(
                f"https://api.bilibili.com/x/web-interface/popular?ps={ps}&pn={pn}")
        except RuntimeError as e:
            print(f"  [warn] 热门列表第 {pn} 页获取失败: {e}")
            continue
        for item in data.get("data", {}).get("list", []):
            videos.append({
                "bvid": item.get("bvid"),
                "title": item.get("title", ""),
                "danmaku_count": item.get("stat", {}).get("danmaku", 0),
            })
        time.sleep(0.5)
    videos.sort(key=lambda v: v["danmaku_count"], reverse=True)
    return videos


def get_main_cid(bvid):
    """视频主分P的 cid;失败返回 None。"""
    try:
        data = http_get_json(
            f"https://api.bilibili.com/x/player/pagelist?bvid={bvid}")
        pages = data.get("data") or []
        return pages[0]["cid"] if pages else None
    except (RuntimeError, KeyError, IndexError, TypeError):
        return None


def fetch_danmaku_xml(cid):
    """下载弹幕池数据,返回 XML bytes。失败抛异常。

    老弹幕池返回明文 XML,迁移后的池返回 raw-deflate 压缩的 XML
    (无 zlib 头,需要 wbits=-15)。
    """
    raw = http_get(f"https://comment.bilibili.com/{cid}.xml")
    if raw[:1] == b"<":
        return raw
    # 依次尝试 raw deflate 和带 zlib 头的 deflate
    for wbits in (-15, 15):
        try:
            return zlib.decompressobj(wbits).decompress(raw)
        except zlib.error:
            continue
    raise RuntimeError(f"cid={cid} 的弹幕数据无法解压(前16字节: {raw[:16].hex()})")


def parse_danmaku_xml(xml_bytes):
    """解析弹幕 XML,返回 [(text, time_s, mode), ...]。

    p 字段: progress(秒),mode,fontsize,color,ctime,pool,user_hash,row_id
    mode: 1-3滚动 4底部 5顶部 6逆向 7高级 8代码 9BAS弹幕
    """
    results = []
    for m in D_PATTERN.finditer(xml_bytes):
        p_fields = m.group(1).decode("ascii", "replace").split(",")
        try:
            time_s = float(p_fields[0]) if p_fields[0] else 0.0
            mode = int(p_fields[1]) if len(p_fields) > 1 else 0
        except ValueError:
            time_s, mode = 0.0, 0
        text = html.unescape(decode_danmaku_text(m.group(2))).strip()
        # 去掉控制字符和零宽字符,避免脏数据
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f​-‏‪-‮﻿]", "", text).strip()
        if text:
            results.append((text, time_s, mode))
    return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="B站弹幕采集")
    parser.add_argument("--target", type=int, default=20000, help="目标弹幕条数(去重后)")
    parser.add_argument("--pages", type=int, default=3, help="抓取热门列表页数(每页50个视频)")
    parser.add_argument("--max-videos", type=int, default=0, help="最多处理多少个视频(0=不限)")
    parser.add_argument("--delay", type=float, default=0.8, help="视频间请求间隔秒数")
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).resolve().parent / "danmaku_raw.jsonl"),
                        help="输出文件")
    parser.add_argument("--append", action="store_true",
                        help="追加模式: 加载已有文件的文本作为去重基准,继续采集到目标总量")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()          # 全局文本去重
    existing = 0
    if args.append and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    seen_base = json.loads(line)["text"]
                    seen.add(seen_base)
                    existing += 1
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"追加模式: 已加载现有 {existing} 条文本作为去重基准")
    elif out_path.exists():
        out_path.unlink()

    remaining = max(0, args.target - existing)
    print(f"目标 {args.target} 条(现有 {existing},需新增 {remaining}) | 输出 {out_path}")
    videos = get_popular_videos(pages=args.pages)
    print(f"热门列表获取到 {len(videos)} 个视频(按弹幕数降序)")

    if args.max_videos:
        videos = videos[:args.max_videos]

    written = 0           # 本次运行新写入的条数
    fetched_total = 0     # 本视频抓到的原始条数(含重复)
    processed = 0         # 实际遍历过的条数(提前达标时会少于抓取数)
    per_video = []

    with open(out_path, "a", encoding="utf-8") as f:
        for idx, v in enumerate(videos, 1):
            if written >= remaining:
                break
            bvid, title = v["bvid"], v["title"]
            cid = get_main_cid(bvid)
            if cid is None:
                print(f"[{idx}/{len(videos)}] {bvid} 无 cid,跳过")
                continue
            try:
                xml_bytes = fetch_danmaku_xml(cid)
                items = parse_danmaku_xml(xml_bytes)
            except RuntimeError as e:
                print(f"[{idx}/{len(videos)}] {bvid} 弹幕获取失败: {e}")
                time.sleep(args.delay)
                continue

            n_new = 0
            for text, time_s, mode in items:
                processed += 1
                if text in seen:
                    continue
                seen.add(text)
                f.write(json.dumps({
                    "text": text,
                    "time_s": round(time_s, 2),
                    "mode": mode,
                    "bvid": bvid,
                    "title": title,
                }, ensure_ascii=False) + "\n")
                n_new += 1
                written += 1
                if written >= remaining:
                    break

            fetched_total += len(items)
            per_video.append((title, len(items), n_new))
            print(f"[{idx}/{len(videos)}] 累计 {existing + written}/{args.target} | "
                  f"本视频 {len(items)} 条(新 {n_new}) | {title[:24]}")

            time.sleep(args.delay + random.uniform(0, 0.4))

    # 汇总
    total_in_file = existing + written
    print("\n===== 采集完成 =====")
    print(f"文件: {out_path} | 总条数: {total_in_file} | 本次新增: {written}")
    print(f"本次抓取(含重复): {fetched_total} | 遍历 {processed} 条 | 本次去重率: "
          f"{1 - written / max(processed, 1):.1%}")
    print(f"本次覆盖视频数: {len(per_video)} | 平均每视频: {written / max(len(per_video), 1):.0f} 条")
    print("\n每视频明细(前10):")
    for title, total, new in per_video[:10]:
        print(f"  {total:>6} 条(新 {new:>6})  {title[:40]}")
    print("\n样本预览(前8条):")
    with open(out_path, encoding="utf-8") as f:
        for i, line in zip(range(8), f):
            item = json.loads(line)
            print(f"  [{item['mode']}] {item['text'][:50]}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断: 已采集的数据已保存。")
        sys.exit(0)
