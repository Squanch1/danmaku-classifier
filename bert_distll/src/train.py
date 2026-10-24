"""
# 该脚本用于: BERT模型的训练 和 评估.
# todo 1.加载配置文件 和 核心工具类.
# todo 2. 定义模型训练函数: 封装完成的训练流程, 例如: 数据加载 -> 模型训练 -> 验证 -> 保存.
# todo 3. 定义模型验证函数
# todo 4. 定义主函数
"""
import torch

from bert.src.config import Config

from utils import build_dataloader
from bert_classifier import TMFBertClassifer
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm  # 进度条
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# todo 1.加载配置文件 和 核心工具类.
conf = Config()


# todo 2. 定义模型训练函数: 封装完成的训练流程, 例如: 数据加载 -> 模型训练 -> 验证 -> 保存.

def model2train(train_dataloader, dev_dataloader, model, device):
    # 1.实例化损失函数和优化器
    loss_f = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=conf.lr)

    # 最优分数和总迭代次数
    best_f1 = 0.0
    total_iters = 0

    # 2.遍历训练轮次和数据加载器
    for epoch in range(conf.epochs):
        total_loss = 0.0  # 当前迭代的总损失
        total_iter_epoch = 0  # 当期迭代的总迭代次数
        start = time.time()
        model.train()

        for i, (input_ids, attention_mask, labels) in \
                tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                     desc=f"模型训练中，当前轮次为：{epoch + 1}"):
            # 3.模型训练 （前向传播-->计算损失-->梯度清零-->反向传播-->参数更新）
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask)
            loss = loss_f(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 4.记录训练监控日志
            total_loss += loss.item()
            total_iters += 1
            total_iter_epoch += 1

            # 每20步打印一次损失
            if (i + 1) % 20 == 0:
                avg_loss = total_loss / total_iter_epoch
                print(f'\nEpoch：{epoch + 1}, Step：{total_iters}, Loss：{avg_loss:4f} ,time: {time.time() - start}s')
                # 保存训练日志
                with open(conf.log_path, 'a', encoding='utf-8') as f:
                    f.write(
                        f'Epoch：{epoch + 1}, Step：{total_iters}, Loss：{avg_loss:4f} ,time: {time.time() - start}s\n')
            # 每500步获取批次模型验证模型，保存最优模型
            if (i + 1) % 500 == 0 or (i + 1) == len(train_dataloader):
                f1, report, confmat = model2eval(dev_dataloader, model, device)
                model.train()
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), conf.best_model)
                print(f'Epoch：{epoch + 1}，step：{total_iters}，f1:{f1:.4f},time: {time.time() - start}s')
                # 保存训练日志
                with open(conf.log_path, "a", encoding="utf-8") as f:
                    f.write(f'\nEpoch：{epoch + 1}，step：{total_iters}，f1:{f1:.4f},time: {time.time() - start}s\n')
                    f.write(f'\n分类评估报告\n:{report}\n')
                    f.write(f'\n混淆矩阵\n:{confmat}\n')
    # 4.模型保存
    torch.save(model.state_dict(), conf.last_model)


# todo 3. 定义模型验证函数
def model2eval(dev_dataloader, model, device):
    model.eval()
    y_pred_list, y_true_list = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dev_dataloader, total=len(dev_dataloader), desc="模型验证中"):
            # 前向传播
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask)
            y_pred = torch.argmax(logits, dim=-1)
            y_pred_list.extend(y_pred.tolist())
            y_true_list.extend(labels.tolist())
    # 4.计算评估指标
    f1 = f1_score(y_true_list, y_pred_list, average="weighted")
    report = classification_report(y_true_list, y_pred_list)
    confmat = confusion_matrix(y_true_list, y_pred_list)
    return f1, report, confmat


# todo 4. 定义主函数
def main():
    # 1. 加载数据和数据预处理
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    # 2.实例化模型
    model = TMFBertClassifer().to(conf.device)
    # 3.模型训练
    model2train(train_dataloader, dev_dataloader, model, conf.device)
    # 4.模型评估
    model_best = TMFBertClassifer().to(conf.device)
    model_best.load_state_dict(torch.load(conf.best_model))
    f1, report, confmat = model2eval(test_dataloader, model_best, conf.device)
    print(f'测试集上模型的f1:{f1:.4f}')
    print(f'测试集上模型的分类评估报告:\n{report}')
    print(f'测试集上模型的混淆矩阵:\n{confmat}')
    # 保存评估结果
    with open(conf.log_path, 'a', encoding='utf-8') as f:
        f.write(f'测试集上模型的f1:{f1:.4f}\n')
        f.write(f'测试集上模型的分类评估报告:\n{report}\n')
        f.write(f'测试集上模型的混淆矩阵:\n{confmat}\n')


if __name__ == '__main__':
    main()
    # print(conf.device)
