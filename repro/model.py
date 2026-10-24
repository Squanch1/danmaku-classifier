"""使用与原训练一致的 BERT pooler + Linear 四分类结构。"""

from pathlib import Path

import torch
from torch import nn
from transformers import BertConfig, BertModel


class BertClassifier(nn.Module):
    def __init__(self, config_path: Path, pretrained: bool = False) -> None:
        super().__init__()
        self.bert = (BertModel.from_pretrained(str(config_path), local_files_only=True) if pretrained
                     else BertModel(BertConfig.from_pretrained(str(config_path), local_files_only=True)))
        self.fc = nn.Linear(self.bert.config.hidden_size, 4)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output)
