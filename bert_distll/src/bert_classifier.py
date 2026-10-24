"""
# 该脚本用于: BERT分类模型的定义 和 创建.
todo 1.创建配置文件对象.
todo 2. 定义BERT分类模型(继承nn.Module), 实现自定义神经网络.
"""
from config import Config
import torch.nn as nn
from transformers import BertModel
conf = Config()

class TMFBertClassifer(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(conf.bert_path)
        # 构建分类输出头  输出神经元为类别数
        self.fc = nn.Linear(conf.hidden_size,conf.class_num)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(output.pooler_output)
        return logits



if __name__ == '__main__':
    from utils import build_dataloader
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    model = TMFBertClassifer()
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
        print(model(input_ids, attention_mask).shape)
        break

