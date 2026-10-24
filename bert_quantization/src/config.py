"""
该脚本用于: BERT版弹幕质量分级 配置文件.
# todo 1.源文件路径
# 训练集路径、测试集路径、验证集路径、标签类别文件路径、标签类别、日志保存路径
# todo 2.模型保存路径
# todo 3.bert模型相关配置
# 模型名称、模型文件路径、隐藏层的维度
# todo 4. 训练参数
# 类别数量、训练轮次、批次大小、最大句子长度 、学习率
# todo 5.设备配置
"""
import time
from pathlib import Path

import torch
from transformers import BertTokenizer

# 锚定 bert/ 目录, 任何位置运行都不会路径错乱
BERT_DIR = Path(__file__).resolve().parent.parent


class Config:
    def __init__(self):
        # todo 1.源文件路径(JSONL 格式: {"text": ..., "label": 1~4})
        self.train_data_path = BERT_DIR / 'processed_data' / 'train.jsonl'
        self.val_data_path = BERT_DIR / "processed_data" / "val.jsonl"
        self.test_data_path = BERT_DIR / "processed_data" / "test.jsonl"
        self.class_path = BERT_DIR / "raw_data" / "class.txt"

        # 标签类别名(取每行第一个词: 1级/2级/3级/4级)
        with open(self.class_path, 'r', encoding='utf-8') as f:
            self.class_list = [line.split()[0] for line in f if line.strip()]
        # 日志保存路径
        self.log_path = BERT_DIR / f'log/{time.strftime("%Y_%m_%d")}_log.txt'
        # todo 2.模型保存路径
        self.best_model = BERT_DIR / 'save_model/bert_classifier_best_model.pt'
        self.last_model = BERT_DIR / 'save_model/bert_classifier_last_model.pt'
        # todo 3.bert模型相关配置
        # 模型文件路径、隐藏层的维度
        self.bert_path = str(BERT_DIR / 'bert-base-chinese')
        self.hidden_size = 768
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # 词表大小 用于学生模型的构建
        self.vocab_size = self.tokenizer.vocab_size
        # todo 4. 训练参数
        # 类别数量、训练轮次、批次大小、最大句子长度 、学习率
        self.class_num = len(self.class_list)
        self.epochs = 5
        self.batch_size = 32
        self.max_len = 128        # 弹幕上限 100 字, 128 零截断
        self.lr = 5e-5
        # todo 5.设备配置
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        # todo 6.学生模型配置
        # 词嵌入维度、LSTM隐藏层维度、LSTM层数、Dropout概率
        self.embedding_dim = 128
        self.lstm_hidden_size = 128
        self.num_layer = 2
        self.dropout = 0.2
        # todo 7.蒸馏参数
        self.temperature = 2.0
        self.alpha = 0.5


if __name__ == '__main__':
    config = Config()
    print(config.class_list, config.class_num)
