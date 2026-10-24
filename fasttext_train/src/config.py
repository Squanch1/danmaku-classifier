from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # fasttext_train/ 目录(脚本在 src/ 下)
DATA_DIR = HERE.parent / "data"                 # 项目 data/ 目录


class Config:
    def __init__(self):
        # 1.原始数据路径(jsonl: text / label, 0~3 级)
        # 2026-09-02 重新划分: 测试/验证/训练 全 1:1:1:1 均衡(1级含少量虚拟弹幕), 数据在 bert/processed_data/
        self.train_data_path = HERE.parent / "bert" / "processed_data" / "train.jsonl"     # 4,000 条, 均衡
        self.test_data_path = HERE.parent / "bert" / "processed_data" / "test.jsonl"   # 1,600 条, 均衡
        self.val_data_path = HERE.parent / "bert" / "processed_data" / "val.jsonl"       # 1,000 条, 均衡
        # 2.处理后的数据路径
        self.processed_data_path = HERE / "processed_data"
        # 3.分类名称(1 垃圾 ~ 4 优质)
        self.class_list = ["1", "2", "3", "4"]
        # 4.模型保存路径
        self.save_model_path = HERE / "save_models"
