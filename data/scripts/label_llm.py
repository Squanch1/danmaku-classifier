"""用大模型（OpenAI 兼容接口）批量给弹幕打质量等级标签（四分类: 1垃圾/2低质/3普通/4优质）。

流程:
    1. 读入原始弹幕 JSONL(danmaku_raw.jsonl),排除已标注过的文本
    2. 无监督抽样: 按 bvid 均匀抽 TARGET_N 条(跨视频、跨语境多样)
    3. few-shot 批量标注: 一次送 BATCH_SIZE 条给 LLM,返回 JSON
    4. 断点续标: 每批立即落盘 quality_partial.csv,中断重跑自动跳过已标
    5. 汇总输出 quality_labeled_YYYYMMDD.csv(含标签+理由)

标注规范与 annotate_pilot.py 一致(四级质量方案)。
API Key 留空则自动从 ~/.claude/settings.json 的 DeepSeek 配置读取。
"""

import csv
import json
import math
import os
import re
import time
from datetime import date

import pandas as pd
from openai import OpenAI

# ============================ 配置区 ============================
# OpenAI 兼容接口。常见服务商:
#   DeepSeek    : BASE_URL = "https://api.deepseek.com"                 MODEL = "deepseek-chat"
#   通义千问 Qwen: BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  MODEL = "qwen-plus"
#   智谱 GLM     : BASE_URL = "https://open.bigmodel.cn/api/paas/v4"    MODEL = "glm-4-flash"
#   OpenAI      : BASE_URL = "https://api.openai.com/v1"                MODEL = "gpt-4o-mini"
API_KEY = ""   # ← 留空自动读取; 手动指定可覆盖
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..")
INPUT_JSONL = os.path.join(DATA_DIR, "danmaku_raw.jsonl")   # 要标注的原始弹幕
TARGET_N = 500    # 本次测试标注 500 条
BATCH_SIZE = 30   # 每批送多少条给 LLM
SEED = 42         # 抽样随机种子,保证可复现
# 已标注文件(跳过其中的文本,避免重复标注)
EXCLUDE_FILES = [
    os.path.join(DATA_DIR, "pilot_labeled.jsonl"),
    os.path.join(DATA_DIR, "candidates_high_labeled.jsonl"),
]
# ================================================================

LABELS = [1, 2, 3, 4]
HEADER_OUT = ["内容", "视频标题", "bvid", "出现时间秒", "模式", "标签", "理由"]

# DeepSeek 计价(元 / 百万 tokens)——deepseek-chat 已映射为 deepseek-v4-flash 非思考模式,
# 以官方 2026-04-26 起生效价为准: 缓存命中 0.02 / 未命中 1 / 输出 2
PRICE_IN = 1.0          # 输入(未命中缓存)
PRICE_IN_CACHED = 0.02  # 输入(命中缓存,长 system prompt 大部分会命中)
PRICE_OUT = 2.0         # 输出


def partial_path() -> str:
    return os.path.join(HERE, "quality_partial.csv")


SYSTEM_PROMPT = (
    "你是弹幕质量标注专家。对用户给出的每条弹幕标注质量等级,输出 JSON: "
    '{"id": 编号, "label": <1-4的整数>, "reason": "<一句话理由,不超过30字>"}。\n\n'
    "等级定义:\n"
    "- 1级(垃圾/违规): 广告引流、卖片、联系方式/网址、辱骂人身攻击、低俗色情、政治敏感\n"
    "- 2级(低质): 无信息量的灌水,如纯\"哈哈哈哈\"、纯表情颜文字、\"打卡\"、\"+1\"、无意义数字串、与视频内容完全无关的水评\n"
    "- 3级(普通): 正常互动与情绪表达,如\"666\"、\"前方高能\"、剧情/画面吐槽、队形参与、简单提问或回应、情感共鸣\n"
    "- 4级(优质/精华): 有信息量到高价值,包括剧情细节补充、有观点的相关讨论、完整句子的外文翻译、科普解释与纠错、"
    "BGM/出处提示、关键时间点导航(带时间戳)、独到见解或创造梗\n\n"
    "边界裁决规则:\n"
    "1. 纯\"哈哈哈哈\"/纯表情,无论多长 = 2级\n"
    "2. \"哈哈哈\"+有内容的吐槽 = 3级\n"
    "3. 简短常见互动(\"666\"、\"前方高能\"、\"名场面\"、\"泪目\") = 3级\n"
    "4. 完整句子以上的翻译 = 4级,不再区分句子数量\n"
    "5. 指出视频中错误并给出正确信息 = 4级\n"
    "6. 广告/联系方式/网址 = 1级,无例外\n"
    "7. 与视频无关的灌水 = 2级;表达对视频/up主的情绪 = 3级\n"
    "8. 拿不准时取保守值 3 级\n\n"
    "Few-shot 示例:\n"
    "\"哈哈哈哈哈哈\" -> {\"id\": 0, \"label\": 2, \"reason\": \"纯情绪无信息量\"}\n"
    "\"前方高能预警\" -> {\"id\": 0, \"label\": 3, \"reason\": \"常见互动提示\"}\n"
    "\"66666\" -> {\"id\": 0, \"label\": 3, \"reason\": \"常见赞赏互动\"}\n"
    "\"加V看片:xxxxx\" -> {\"id\": 0, \"label\": 1, \"reason\": \"广告引流\"}\n"
    "\"这里翻译:他说的是'我们会永远在一起'\" -> {\"id\": 0, \"label\": 4, \"reason\": \"完整一句翻译\"}\n"
    "\"科普:视频里把第二定律讲错了,正确的表述是……\" -> {\"id\": 0, \"label\": 4, \"reason\": \"纠错科普\"}\n"
    "\"up主说的对,这里其实是致敬了XX电影\" -> {\"id\": 0, \"label\": 4, \"reason\": \"补充相关知识\"}\n"
    "\"打卡\" -> {\"id\": 0, \"label\": 2, \"reason\": \"无信息量灌水\"}\n"
    "\"淦,这也太帅了吧\" -> {\"id\": 0, \"label\": 3, \"reason\": \"情绪表达\"}\n\n"
    "只输出 JSON 数组,不要输出其他内容。"
)


def build_user_prompt(items):
    """把一批弹幕拼成带编号的 prompt。"""
    lines = "\n".join(f"{i}. {t}" for i, t in enumerate(items))
    return (
        f"判断下面 {len(items)} 条弹幕的质量等级,按编号顺序返回 JSON 数组,"
        f'每项形如 {{"id": 编号, "label": 1-4的整数, "reason": "理由"}},不要输出其他内容:\n\n{lines}'
    )


def parse_reply(text, n):
    """解析 LLM 返回的 JSON,校验 id 与 label,返回按序的 (label, reason) 列表。"""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)   # 去掉可能包裹的 markdown 代码块
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("回复里没有 JSON 数组")
    arr = json.loads(t[start:end + 1])
    by_id = {int(item["id"]): item for item in arr}
    result = []
    for i in range(n):
        item = by_id.get(i, {})
        lab = item.get("label")
        if isinstance(lab, str):
            m = re.search(r"[1-4]", lab)
            lab = int(m.group(0)) if m else None
        if lab not in LABELS:
            raise ValueError(f"第 {i} 条标签非法:{lab!r}")
        result.append((lab, str(item.get("reason", ""))[:100]))
    return result


def label_batch(client, items, retries=3):
    """给一批弹幕打标签,失败重试。返回 (标签列表, usage 字典)。"""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(items)},
                ],
                temperature=0,
            )
            text = resp.choices[0].message.content
            usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
            return parse_reply(text, len(items)), usage
        except Exception as e:
            print(f"    批次标注失败(第 {attempt + 1}/{retries} 次):{e}")
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    return None, {}   # 全部重试失败


def sample_uniform(df, target_n, seed):
    """按 bvid 均匀抽样:每个视频抽 ceil(target_n / 视频数),不足则全取。"""
    groups = list(df.groupby("bvid"))
    per = max(1, math.ceil(target_n / len(groups)))
    parts = []
    for _, g in groups:
        parts.append(g.sample(min(per, len(g)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def resolve_api_key():
    """API Key 优先级: 配置区 > 环境变量 > ~/.claude/settings.json。"""
    if API_KEY.strip():
        return API_KEY.strip()
    for env_name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        v = os.environ.get(env_name, "")
        if v:
            return v
    settings = os.path.join(os.path.expanduser("~"), ".claude", "settings.json")
    try:
        with open(settings, encoding="utf-8") as f:
            return json.load(f).get("env", {}).get("ANTHROPIC_AUTH_TOKEN", "")
    except (OSError, json.JSONDecodeError):
        return ""


def load_excluded_texts():
    """已标注文本集合(自身断点文件 + 其他已标注 jsonl)。"""
    texts = set()
    p = partial_path()
    if os.path.exists(p):
        with open(p, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                texts.add(row["内容"])
    for f_path in EXCLUDE_FILES:
        if not os.path.exists(f_path):
            continue
        for line in open(f_path, encoding="utf-8"):
            try:
                texts.add(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                pass
    return texts


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    api_key = resolve_api_key()
    if not api_key:
        print("未找到 API Key: 请填脚本配置区 API_KEY,或设置环境变量。")
        return

    df = pd.read_json(INPUT_JSONL, lines=True)
    print(f"读入 {len(df)} 条弹幕")

    done_texts = load_excluded_texts()
    if done_texts:
        print(f"排除已标注 {len(done_texts)} 条(自身断点 + 既有标注文件)")

    # 无监督抽样(排除已标注的),抽够 TARGET_N
    pool = df[~df["text"].isin(done_texts)]
    sampled = sample_uniform(pool, TARGET_N, SEED).head(TARGET_N)
    print(f"本次待标注 {len(sampled)} 条,来自 {sampled['bvid'].nunique()} 个视频")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    p = partial_path()
    new_file = not os.path.exists(p)
    pf = open(p, "a", encoding="utf-8", newline="")
    writer = csv.writer(pf)
    if new_file:
        writer.writerow(HEADER_OUT)

    n = len(sampled)
    usage_sum = {"prompt_tokens": 0, "completion_tokens": 0,
                 "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}
    try:
        for start in range(0, n, BATCH_SIZE):
            chunk = sampled.iloc[start:start + BATCH_SIZE]
            items = chunk["text"].astype(str).tolist()
            labels, usage = label_batch(client, items)
            if labels is None:
                print(f"    [{start}/{n}] 标注失败,跳过本批")
                continue
            for k in usage_sum:
                usage_sum[k] += usage.get(k, 0)
            for (_, row), (lab, reason) in zip(chunk.iterrows(), labels):
                writer.writerow([
                    row["text"], row.get("title", ""), row["bvid"],
                    row.get("time_s", ""), row.get("mode", ""), lab, reason,
                ])
            pf.flush()
            print(f"    [{start + len(chunk)}/{n}] 已标 {start + len(chunk)} 条")
            time.sleep(0.5)   # 温和限速
    finally:
        pf.close()

    # 汇总输出(以「内容」去重,保留首次标注)
    rows = []
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    seen, uniq = set(), []
    for row in rows:
        if row["内容"] in seen:
            continue
        seen.add(row["内容"])
        uniq.append(row)

    out = os.path.join(HERE, f"quality_labeled_{date.today().strftime('%Y%m%d')}.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER_OUT)
        for row in uniq:
            w.writerow([row[k] for k in HEADER_OUT])

    from collections import Counter
    dist = Counter(row["标签"] for row in uniq)
    print(f"\n标注完成: 共 {len(uniq)} 条,已保存到 {out}")
    print("等级分布:", dict(sorted(dist.items())))

    # token 用量与费用
    hit = usage_sum["prompt_cache_hit_tokens"]
    miss = usage_sum["prompt_cache_miss_tokens"]
    if hit == 0 and miss == 0:          # 兼容不区分缓存的返回
        miss = usage_sum["prompt_tokens"]
    out_t = usage_sum["completion_tokens"]
    cost = (hit * PRICE_IN_CACHED + miss * PRICE_IN + out_t * PRICE_OUT) / 1_000_000
    print(f"\n本次 token 用量: 输入 {miss + hit:,}(缓存命中 {hit:,}) | 输出 {out_t:,}")
    print(f"估算费用: ¥{cost:.4f}(按 deepseek-v4-flash 官方价)")


if __name__ == "__main__":
    main()
