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

# 锚定 bert_distll/ 目录, 任何位置运行都不会路径错乱
DISTLL_DIR = Path(__file__).resolve().parent.parent
# 数据文件复用 bert/processed_data(划分好的 jsonl, 标签 0~3)
DATA_DIR = DISTLL_DIR.parent / "bert" / "processed_data"


class Config:
    def __init__(self):
        # todo 1.源文件路径(JSONL 格式: {"text": ..., "label": 0~3})
        self.train_data_path = DATA_DIR / 'train.jsonl'
        self.val_data_path = DATA_DIR / "val.jsonl"
        self.test_data_path = DATA_DIR / "test.jsonl"
        self.class_path = DISTLL_DIR.parent / "bert" / "raw_data" / "class.txt"

        # 标签类别名(取每行第一个词: 1级/2级/3级/4级)
        with open(self.class_path, 'r', encoding='utf-8') as f:
            self.class_list = [line.split()[0] for line in f if line.strip()]
        # 日志保存路径
        self.log_path = DISTLL_DIR / f'log/{time.strftime("%Y_%m_%d")}_log.txt'
        # todo 2.模型保存路径(bert_distll/save_model/ 下, 避免与 bert 训练模型冲突)
        self.best_model = DISTLL_DIR / 'save_model/bilstm_best_model.pt'
        self.last_model = DISTLL_DIR / 'save_model/bilstm_last_model.pt'
        # todo 3.bert模型相关配置
        # 模型文件路径、隐藏层的维度
        self.bert_path = str(DISTLL_DIR / 'bert-base-chinese')
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
        self.embedding_dim = 256
        self.lstm_hidden_size = 256
        self.num_layer = 2
        self.dropout = 0.2
        # todo 7.蒸馏参数
        self.temperature = 4.0
        self.alpha = 0.7
        # todo 8.蒸馏路径: 老师模型(BERT best) + 学生模型保存
        self.teacher_best_model = DISTLL_DIR.parent / "bert" / "save_model" / "bert_classifier_best_model.pt"
        self.student_best_model = DISTLL_DIR / 'save_model/bilstm_distill_best_model.pt'
        self.student_last_model = DISTLL_DIR / 'save_model/bilstm_distill_last_model.pt'


if __name__ == '__main__':
    config = Config()
    print(config.class_list, config.class_num)
