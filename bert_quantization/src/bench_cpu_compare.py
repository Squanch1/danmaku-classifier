"""
CPU 推理基准: fp32(原版) vs qint8(量化) 同赛道对比。

回答: "无 GPU 服务器上, 量化值不值?"
  - 同一台 CPU、同一批测试数据, 对比 fp32 和 qint8 的速度与精度
  - fp16 是 GPU 赛道, 不参与本对比

用法: python bert_quantization/src/bench_cpu_compare.py [--n 1000 --batch 32]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

BERT_SRC = Path(__file__).resolve().parent.parent.parent / "bert" / "src"
sys.path.insert(0, str(BERT_SRC))

from config import Config            # noqa: E402
from utils import build_dataloader   # noqa: E402
from bert_classifier import TMFBertClassifer  # noqa: E402

conf = Config()
QUANT_MODEL = Path(__file__).resolve().parent.parent / "save_model" / "bert_classifier_qint8_model.pt"
FP32_MODEL = BERT_SRC.parent / "save_model" / "bert_classifier_best_model.pt"

CPU = torch.device("cpu")


def time_predict(model, dataloader, desc):
    """在 CPU 上推理整个 dataloader, 返回 (总秒数, accuracy)。"""
    model.eval()
    correct = total = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == labels).sum().item()
            total += len(labels)
    elapsed = time.perf_counter() - t0
    acc = correct / total
    print(f"{desc}: {total} 条 | 总耗时 {elapsed:.1f}s | 单条 {elapsed / total * 1000:.1f} ms | accuracy {acc:.4f}")
    return elapsed, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="测试条数(0=全部)")
    args = parser.parse_args()

    # 用 build_dataloader 的 val(1000 条均衡, CPU 上测速适中)
    _, dev_dataloader, _ = build_dataloader()

    # 1. fp32 原版(CPU)
    fp32 = TMFBertClassifer().to(CPU)
    fp32.load_state_dict(torch.load(FP32_MODEL, map_location=CPU))
    print(f"fp32 模型已加载: {FP32_MODEL}")

    # 2. qint8 量化(CPU)
    q8 = torch.load(QUANT_MODEL, map_location=CPU, weights_only=False)
    print(f"qint8 模型已加载: {QUANT_MODEL}")

    e1, a1 = time_predict(fp32, dev_dataloader, "fp32 (CPU)")
    e2, a2 = time_predict(q8, dev_dataloader, "qint8(CPU)")

    print("\n===== 对比结论 =====")
    print(f"速度提升: qint8 比 fp32 快 {e1 / e2:.2f} 倍")
    print(f"精度变化: fp32 {a1:.4f} -> qint8 {a2:.4f} (Δ{a2 - a1:+.4f})")


if __name__ == "__main__":
    main()
