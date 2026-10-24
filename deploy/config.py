"""
部署配置: 自包含, 不依赖任何训练代码。
模型权重 / tokenizer / 类别文件都在 deploy/model/ 下。
"""
from pathlib import Path

import torch

DEPLOY_DIR = Path(__file__).resolve().parent
MODEL_DIR = DEPLOY_DIR / "model"


class Config:
    def __init__(self):
        # fp16 权重 (GPU 上转 half 加速; CPU 上加载时自动转回 float32 运行)
        self.model_path = MODEL_DIR / "bert_classifier_fp16_model.pt"
        # tokenizer + 架构配置 (与训练一致: bert-base-chinese)
        self.bert_path = str(MODEL_DIR / "bert-base-chinese")
        # 类别文件: 每行第一个词即等级名 (1级/2级/3级/4级)
        with open(MODEL_DIR / "class.txt", "r", encoding="utf-8") as f:
            self.class_list = [line.split()[0] for line in f if line.strip()]
        self.class_num = len(self.class_list)
        # 弹幕上限 100 字, 128 零截断 (与训练一致)
        self.max_len = 128
        # 动态批处理参数: 攒够 batch_size 或等待 max_wait 秒后一次性前向
        self.batch_size = 32
        self.max_wait = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    c = Config()
    print(c.class_list, c.class_num, c.device)
