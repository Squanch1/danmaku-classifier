#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 fastText 模型评估: 在测试集上算指标, 输出对齐的混淆矩阵, 保存报告。

输出(保存在 训练结果/ 目录下):
  report_<模型名>.txt   指标 + 对齐混淆矩阵(人类可读)
  report_<模型名>.json  结构化指标

用法:
  python evaluate_model.py --model save_models/fasttext_char_auto.bin --char
  python evaluate_model.py --model save_models/fasttext_word_default.bin --word
"""

import argparse
import json
import sys
from pathlib import Path

import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from config import Config

config = Config()
# 2026-09-02 数据改为 0~3 标签(bert/processed_data), 评估按 0~3, 报告显示 1~4
LABELS = [0, 1, 2, 3]


def load_test(use_word: bool):
    name = "test_word.csv" if use_word else "test_char.csv"
    df = pd.read_csv(config.processed_data_path / name, sep="\t")
    y_true = [int(l.replace("__label__", "")) for l in df["label"]]
    return df["text"].tolist(), y_true


def parse_pred(text):
    return int(text.rsplit("__label__", 1)[1])


def fmt_cm(cm, labels):
    """混淆矩阵对齐格式化, 保证每行不换行、列宽一致。标签显示为 1~4(内部 0~3)。"""
    show = [lab + 1 for lab in labels]
    # 对齐需要预扫描最大数字宽度
    width = max(len(str(v)) for row in cm for v in row)
    head = "真实\\预测".ljust(width + 2) + "".join(f"{lab:>{width + 2}}" for lab in show)
    lines = [head]
    for i, row in enumerate(cm):
        lines.append(f"   {show[i]}".ljust(width + 2) + "".join(f"{v:>{width + 2}}" for v in row))
    return "\n".join(lines)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="fastText 模型评估")
    parser.add_argument("--model", required=True, help="模型路径, 如 save_models/fasttext_char_auto.bin")
    parser.add_argument("--word", action="store_true", help="使用词分词测试集(默认字符)")
    args = parser.parse_args()

    model = fasttext.load_model(args.model)
    texts, y_true = load_test(args.word)
    y_pred = [parse_pred(model.predict(t, k=1)[0][0]) for t in texts]

    # 指标
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    mae = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    report = []
    report.append("=" * 50)
    report.append(f"模型: {args.model}")
    report.append(f"测试集: {len(texts)} 条(按{'词' if args.word else '字符'}切分)")
    report.append("=" * 50)
    report.append(f"accuracy : {acc:.3f}")
    report.append(f"macro-F1 : {np.mean(f1):.3f}")
    report.append(f"weighted-F1: {np.mean(f1):.3f}")
    report.append(f"MAE      : {mae:.2f}")
    report.append("")
    report.append("各类别指标:")
    report.append(f"  {'等级':<4}{'精确率':>8}{'召回率':>8}{'F1':>8}{'样本数':>8}")
    for lab, pi, ri, fi, s in zip(LABELS, p, r, f1, support):
        report.append(f"  {lab + 1:<4}{pi:>8.3f}{ri:>8.3f}{fi:>8.3f}{s:>8}")
    report.append("")
    report.append("混淆矩阵(行=真实, 列=预测):")
    report.append(fmt_cm(cm, LABELS))
    report.append("")
    report.append("=" * 50)
    body = "\n".join(report)
    print(body)

    # 保存
    out_dir = config.save_model_path.parent / "训练结果"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model).stem
    (out_dir / f"report_{model_name}.txt").write_text(body, encoding="utf-8")
    json.dump({"model": args.model, "accuracy": acc,
               "macro_f1": float(np.mean(f1)), "mae": mae,
               "per_class": {lab: {"precision": float(pi), "recall": float(ri),
                                   "f1": float(fi), "support": int(s)}
                             for lab, pi, ri, fi, s in zip(LABELS, p, r, f1, support)},
               "confusion_matrix": cm.tolist()},
              open(out_dir / f"report_{model_name}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n报告已保存: {out_dir}")


if __name__ == "__main__":
    main()
