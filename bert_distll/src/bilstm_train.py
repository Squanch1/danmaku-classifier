"""
双向 LSTM 完整流程: 训练 -> 验证选优 -> 保存 -> 测试集评估 -> 日志记录。

- 模型: student_model.BILSTMClassifier(双向 LSTM, 均值池化)
- 数据: bert/processed_data jsonl(标签 0~3, 均衡分布)
- 保存: bert_distll/save_model/bilstm_{best,last}_model.pt
- 日志: bert_distll/log/bilstm_train_<日期>.txt

用法: python bert_distll/src/bilstm_train.py [--epochs 30 --lr 1e-3 --batch-size 32]
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm

from config import Config
from utils import build_dataloader
from student_model import BILSTMClassifier

conf = Config()
DISTLL_DIR = Path(__file__).resolve().parent.parent


# ---------------- 训练 ----------------

def model2train(train_dataloader, dev_dataloader, model, device, log_f):
    """完整训练循环: 每 100 步在验证集上评估, 按 weighted_f1 保存最优模型。"""
    loss_f = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=conf.lr)
    best_f1 = 0.0
    total_iters = 0

    for epoch in range(conf.epochs):
        total_loss = 0.0
        total_iter_epoch = 0
        start = time.time()
        model.train()

        for i, (input_ids, attention_mask, labels) in \
                tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                     desc=f"BiLSTM 训练中, 当前轮次: {epoch + 1}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_f(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_iters += 1
            total_iter_epoch += 1

            # 每 20 步打印损失
            if (i + 1) % 20 == 0:
                avg_loss = total_loss / total_iter_epoch
                msg = f'Epoch:{epoch + 1}, Step:{total_iters}, Loss:{avg_loss:.4f}'
                print(msg)
                log_f.write(msg + '\n')

            # 每 100 步验证, 保存最优
            if (i + 1) % 100 == 0 or (i + 1) == len(train_dataloader):
                f1, report, confmat = model2eval(dev_dataloader, model, device)
                model.train()
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), conf.best_model)
                msg = f'Epoch:{epoch + 1}, Step:{total_iters}, weighted_f1:{f1:.4f}, 耗时:{time.time() - start:.1f}s'
                print(msg)
                log_f.write(msg + '\n')
                log_f.write(f'验证集报告:\n{report}\n{confmat}\n')
                log_f.flush()

    # 保存最后一轮模型
    torch.save(model.state_dict(), conf.last_model)
    print(f"best_f1: {best_f1:.4f} | best: {conf.best_model} | last: {conf.last_model}")


# ---------------- 评估 ----------------

def model2eval(dataloader, model, device):
    """测试/验证集评估: weighted_f1 + 分类报告 + 混淆矩阵。"""
    model.eval()
    y_pred_list, y_true_list = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, total=len(dataloader), desc="模型评估中"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask)
            y_pred = torch.argmax(logits, dim=-1)
            y_pred_list.extend(y_pred.tolist())
            y_true_list.extend(labels.tolist())

    f1 = f1_score(y_true_list, y_pred_list, average="weighted")
    report = classification_report(y_true_list, y_pred_list)
    confmat = confusion_matrix(y_true_list, y_pred_list)
    return f1, report, confmat


# ---------------- 主流程 ----------------

def main():
    parser = argparse.ArgumentParser(description="BiLSTM 完整训练流程")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮次(LSTM 从零学, 建议 30+)")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率(LSTM 建议 1e-3 量级)")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    args = parser.parse_args()

    conf.epochs = args.epochs
    conf.lr = args.lr
    conf.batch_size = args.batch_size

    device = conf.device
    print(f"设备: {device} | epochs={args.epochs} | lr={args.lr} | batch={args.batch_size}")

    # 1.加载数据
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    print(f"训练 {len(train_dataloader.dataset)} | 验证 {len(dev_dataloader.dataset)} | 测试 {len(test_dataloader.dataset)}")

    # 2.创建双向 LSTM(随机初始化)
    model = BILSTMClassifier().to(device)
    print(f"BiLSTM: Embedding({conf.vocab_size}->{conf.embedding_dim}) "
          f"+ BiLSTM({conf.lstm_hidden_size}×{conf.num_layer}层双向) + Linear->{conf.class_num}")

    # 3.日志文件
    (DISTLL_DIR / "save_model").mkdir(parents=True, exist_ok=True)
    (DISTLL_DIR / "log").mkdir(parents=True, exist_ok=True)
    log_path = DISTLL_DIR / "log" / f"bilstm_train_{datetime.now():%Y%m%d_%H%M}.txt"
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"BiLSTM 训练日志 | {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        log_f.write(f"设备: {device} | epochs={args.epochs} | lr={args.lr} | batch={args.batch_size}\n")

        # 4.训练
        model2train(train_dataloader, dev_dataloader, model, device, log_f)

        # 5.加载最优模型, 测试集评估
        model_best = BILSTMClassifier().to(device)
        model_best.load_state_dict(torch.load(conf.best_model, map_location=device))
        f1, report, confmat = model2eval(test_dataloader, model_best, device)
        msg = f'\n[测试集] BiLSTM weighted_f1:{f1:.4f}\n报告:\n{report}\n混淆矩阵:\n{confmat}\n'
        print(msg)
        log_f.write(msg)

    print(f"日志已保存: {log_path}")


if __name__ == "__main__":
    main()
