"""
知识蒸馏: 用 BERT 老师模型(软标签)蒸馏训练 BiLSTM 学生模型。

- 老师: bert_classifier.TMFBertClassifer(BERT, 加载 bert/save_model 的 best)
- 学生: student_model.BILSTMClassifier(双向 LSTM)
- 损失: alpha * T^2 * KL(学生软输出, 老师软输出) + (1-alpha) * CE(学生, 硬标签)
- 保存: bert_distll/save_model/bilstm_distill_{best,last}_model.pt
- 日志: bert_distll/log/distill_<时间戳>.txt

用法: python bert_distll/src/model_distll.py [--epochs 30 --lr 1e-3 --batch-size 32]
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")   # 无显示环境也能保存图片
import matplotlib.pyplot as plt

from config import Config
from utils import build_dataloader
from bert_classifier import TMFBertClassifer
from student_model import BILSTMClassifier
from train import model2eval

conf = Config()
DISTLL_DIR = Path(__file__).resolve().parent.parent


# todo 1.蒸馏训练函数
def model2distill(train_dataloader, dev_dataloader, teacher_model, student_model, device, log_f):
    # 1.构建损失函数和优化器
    soft_loss_f = nn.KLDivLoss(reduction="batchmean")
    hard_loss_f = nn.CrossEntropyLoss()
    optimizer = AdamW(student_model.parameters(), lr=conf.lr)

    # 获取蒸馏参数
    T = conf.temperature
    alpha = conf.alpha
    # 总迭代次数、损失列表
    total_iters = 0
    start_time = time.time()
    loss_list = []
    iter_list = []
    best_f1 = 0.0
    total_loss_sum = 0.0
    total_iters = 0

    for epoch in range(conf.epochs):
        teacher_model.eval()     # 老师模型固定(不训练)
        student_model.train()    # 学生模型训练
        for i, (input_ids, attention_mask, labels) in tqdm(enumerate(train_dataloader),
                                                           total=len(train_dataloader),
                                                           desc=f"模型蒸馏中, 轮次: {epoch + 1}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 1.老师模型前向(不更新梯度)
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask)
            # 2.学生模型前向
            student_logits = student_model(input_ids, attention_mask)

            # 3.计算损失
            # 3.1 软标签损失: 学生软输出 vs 老师软输出(温度 T 软化)
            soft_loss = soft_loss_f(torch.log_softmax(student_logits / T, dim=-1),
                                    torch.softmax(teacher_logits / T, dim=-1))
            # 3.2 硬标签损失: 学生 vs 真实标签
            hard_loss = hard_loss_f(student_logits, labels)
            # 3.3 总损失(乘 T^2 补偿 KL 中温度导致的梯度缩小)
            total_loss = alpha * soft_loss * T * T + (1 - alpha) * hard_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            total_iters += 1

            # 每 10 步打印损失
            if (i + 1) % 10 == 0 :
                avg_loss = total_loss_sum / total_iters
                msg = f'\nEpoch:{epoch + 1}, Step:{total_iters}, Loss:{avg_loss:.4f}, 耗时:{time.time() - start_time:.1f}s'
                print(msg)
                log_f.write(msg + '\n')
                loss_list.append(avg_loss)
                iter_list.append(total_iters)

            # 每 30 步评估, 保存最优学生模型
            if (i + 1) % 30 == 0 :
                f1, report, confmat = model2eval(dev_dataloader, student_model, device)
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(student_model.state_dict(), conf.student_best_model)
                    msg = f'\nEpoch:{epoch + 1}, Step:{total_iters}, weighted_f1:{f1:.4f}'
                else:
                    msg = f'\nEpoch:{epoch + 1}, Step:{total_iters}, weighted_f1:{f1:.4f}'
                print(msg)
                log_f.write(msg)
                log_f.write(f'验证集报告:\n{report}\n{confmat}\n')
                log_f.flush()
                student_model.train()

    # 保存最后一轮学生模型
    torch.save(student_model.state_dict(), conf.student_last_model)
    print(f'蒸馏完成, best_f1:{best_f1:.4f}')

    # 绘制损失曲线
    plt.plot(iter_list, loss_list)
    plt.title('Distill Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    loss_png = DISTLL_DIR / "log" / f"distill_loss_{datetime.now():%Y%m%d_%H%M}.png"
    plt.savefig(str(loss_png))
    plt.close()
    print(f'损失曲线已保存: {loss_png}')


# todo 2.主函数
def main():
    parser = argparse.ArgumentParser(description="BERT -> BiLSTM 知识蒸馏")
    parser.add_argument("--epochs", type=int, default=10, help="蒸馏轮次")
    parser.add_argument("--lr", type=float, default=5e-3, help="学生模型学习率")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    args = parser.parse_args()

    conf.epochs = args.epochs
    conf.lr = args.lr
    conf.batch_size = args.batch_size

    device = conf.device
    print(f"设备: {device} | epochs={args.epochs} | lr={args.lr} | T={conf.temperature} | alpha={conf.alpha}")

    # 1.加载数据
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    print(f"训练 {len(train_dataloader.dataset)} | 验证 {len(dev_dataloader.dataset)} | 测试 {len(test_dataloader.dataset)}")

    # 2.加载老师模型(BERT 最优)
    teacher_model = TMFBertClassifer().to(device)
    teacher_model.load_state_dict(torch.load(conf.teacher_best_model, map_location=device))
    teacher_model.eval()
    print(f"老师模型已加载: {conf.teacher_best_model}")

    # 3.创建学生模型(双向 LSTM)
    student_model = BILSTMClassifier().to(device)
    print(f"学生模型已创建: Embedding({conf.vocab_size}->{conf.embedding_dim}) "
          f"+ BiLSTM({conf.lstm_hidden_size}×{conf.num_layer}层双向) + Linear->{conf.class_num}")

    # 4.蒸馏训练
    (DISTLL_DIR / "save_model").mkdir(parents=True, exist_ok=True)
    (DISTLL_DIR / "log").mkdir(parents=True, exist_ok=True)
    log_path = DISTLL_DIR / "log" / f"distill_{datetime.now():%Y%m%d_%H%M}.txt"
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"知识蒸馏日志 | {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        log_f.write(f"老师: BERT(best) | 学生: BiLSTM | T={conf.temperature} | alpha={conf.alpha}\n")
        log_f.write(f"设备: {device} | epochs={args.epochs} | lr={args.lr}\n")
        model2distill(train_dataloader, dev_dataloader, teacher_model, student_model, device, log_f)

        # 5.加载蒸馏后的最优学生模型, 测试集评估
        best_student = BILSTMClassifier().to(device)
        best_student.load_state_dict(torch.load(conf.student_best_model, map_location=device))
        f1_student, report_student, confmat_student = model2eval(test_dataloader, best_student, device)
        msg = f'\n[测试集] 蒸馏后学生(BiLSTM) weighted_f1:{f1_student:.4f}\n报告:\n{report_student}\n混淆矩阵:\n{confmat_student}\n'
        print(msg)
        log_f.write(msg)

        # 6.老师模型同口径评估(对比)
        f1_teacher, report_teacher, confmat_teacher = model2eval(test_dataloader, teacher_model, device)
        msg = f'\n[测试集] 老师(BERT) weighted_f1:{f1_teacher:.4f}\n报告:\n{report_teacher}\n混淆矩阵:\n{confmat_teacher}\n'
        print(msg)
        log_f.write(msg)

    print(f"日志已保存: {log_path}")


if __name__ == '__main__':
    main()
