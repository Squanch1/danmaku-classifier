#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
弹幕四级分类基线: bert-base-chinese 微调(原 5 级已并入 4 级)

- 类别不均衡用逆频率 class weights 加权交叉熵
- 评估: macro-F1(选模型依据)/ weighted-F1 / accuracy / MAE(有序误差)
- 训完在 test 上输出混淆矩阵、各类别 P/R/F1,并保存逐条预测供错误分析

用法: python train_baseline.py [--epochs 5 --batch-size 32 --lr 2e-5]
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# 国内网络优先走 HF 镜像(需在 import transformers 前设置)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

LABELS = [1, 2, 3, 4]
MAX_LEN = 128          # 弹幕最长 100 字, 128 覆盖全部
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"   # 业界中文分类事实标准
SCRIPT_DIR = Path(__file__).resolve().parent   # Roberta/ 目录
DATA_DIR = SCRIPT_DIR.parent / "data"          # 项目 data/
OUTPUT_DIR = SCRIPT_DIR / "baseline_roberta"


# ---------------- 数据集 ----------------

class DanmakuDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer):
        self.rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        texts = [r["text"] for r in self.rows]
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=MAX_LEN,
            return_tensors=None)
        # PyTorch 交叉熵要求标签 0~n-1, 语义标签 1~4 需减 1(报告时再加回)
        self.labels = [r["label"] - 1 for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {k: torch.tensor(v[i]) for k, v in self.encodings.items()} | {
            "labels": torch.tensor(self.labels[i], dtype=torch.long)}


# ---------------- 加权 Trainer ----------------

class WeightedTrainer(Trainer):
    """按类别权重计算交叉熵(处理 5 级/1 级样本稀缺)。"""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels, weight=self.class_weights)
        return (loss, outputs) if return_outputs else loss


# ---------------- 指标 ----------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    p, r, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "eval_f1_macro": f1_macro,
        "eval_f1_weighted": f1_weighted,
        "eval_accuracy": accuracy_score(labels, preds),
        "eval_mae": float(np.mean(np.abs(labels - preds))),
    }


# 0 基标签(内部) <-> 1~4 语义标签(报告)
def to_semantic(idx):
    return idx + 1


# ---------------- 测试集详细评估 ----------------

def evaluate_test(trainer, dataset, out_path="./test_predictions.jsonl"):
    preds = trainer.predict(dataset)
    y_true = dataset.labels                    # 0 基
    y_pred = np.argmax(preds.predictions, axis=1)

    INTERNAL = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=INTERNAL)
    print("\n===== 测试集评估 =====")
    print("混淆矩阵(行=真实, 列=预测):")
    print("       " + "".join(f"  预测{to_semantic(lab)}" for lab in INTERNAL))
    for lab, row in zip(INTERNAL, cm):
        print(f"  真实{to_semantic(lab)}: " + "".join(f"{v:>7}" for v in row))

    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=INTERNAL, zero_division=0)
    print("\n各类别指标:")
    print(f"  {'等级':<4}{'精确率':>8}{'召回率':>8}{'F1':>8}{'样本数':>8}")
    for lab, pi, ri, fi, s in zip(INTERNAL, p, r, f1, support):
        print(f"  {to_semantic(lab):<4}{pi:>8.3f}{ri:>8.3f}{fi:>8.3f}{s:>8}")
    acc = accuracy_score(y_true, y_pred)
    mae = float(np.mean(np.abs(np.array(y_true) - y_pred)))
    print(f"\naccuracy: {acc:.3f} | macro-F1: {np.mean(f1):.3f} | MAE: {mae:.2f}")

    # 保存逐条预测(语义标签 1~4), 供错误分析
    with open(out_path, "w", encoding="utf-8") as f:
        for row, pred in zip(dataset.rows, y_pred):
            f.write(json.dumps({"text": row["text"], "label": row["label"],
                                "pred": to_semantic(int(pred))}, ensure_ascii=False) + "\n")
    print(f"逐条预测已保存: {out_path}")
    return {"accuracy": acc, "macro_f1": float(np.mean(f1)), "mae": mae,
            "confusion_matrix": cm.tolist()}


# ---------------- 主流程 ----------------

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="弹幕分级基线训练")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"模型: {MODEL_NAME} | GPU: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS))

    # 新评估体系: 训练=富集全量, 验证/测试=自然分布
    train_ds = DanmakuDataset(DATA_DIR / "train_data.jsonl", tokenizer)
    val_ds = DanmakuDataset(DATA_DIR / "val_natural.jsonl", tokenizer)
    test_ds = DanmakuDataset(DATA_DIR / "test_natural.jsonl", tokenizer)
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    # 逆频率类别权重(5级/1级样本稀缺, 加大其梯度贡献)
    counts = Counter(train_ds.labels)
    weights = [len(train_ds) / (len(LABELS) * counts.get(lab, 1)) for lab in LABELS]
    print("class weights:", {lab: round(w, 2) for lab, w in zip(LABELS, weights)})
    class_weights = torch.tensor(weights, dtype=torch.float).cuda()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=100,   # transformers 5.x 已移除 warmup_ratio, 用固定步数(约总步数10%)
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=10,
        seed=args.seed,
        report_to=[],
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\n训练配置: epochs={args.epochs} batch={args.batch_size} lr={args.lr}")
    trainer.train()

    result = evaluate_test(trainer, test_ds, out_path=SCRIPT_DIR / "test_predictions.jsonl")
    (SCRIPT_DIR / "baseline_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n基线训练完成。评估结果已保存: baseline_eval.json")


if __name__ == "__main__":
    main()
