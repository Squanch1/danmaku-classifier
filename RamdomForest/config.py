# 配置文件
from pathlib import Path

HERE = Path(__file__).resolve().parent          # RamdomForest/ 目录
DATA_DIR = HERE.parent / "data"                 # 项目 data/ 目录


class Config:
    def __init__(self):
        # TODO 1.原始文件路径(标注划分, jsonl 格式: text / label, 1~4 级)
        self.train_data_path = DATA_DIR / "test.jsonl"
        self.test_data_path = DATA_DIR / "train.jsonl"
        self.dev_data_path = DATA_DIR / "val.jsonl"
        # TODO 2.预处理后的文件路径
        self.processed_train_data_path = HERE / "data" / "processed_train.csv"
        self.processed_test_data_path = HERE / "data" / "processed_test.csv"
        self.processed_dev_data_path = HERE / "data" / "processed_dev.csv"
        # TODO 3.停用词路径(可选, 不存在时用空表)
        self.stopwords_path = DATA_DIR / "stopwords.txt"
        # TODO 4.标签列表(1 垃圾 ~ 4 优质)
        self.class_list = ["1", "2", "3", "4"]

        # TODO 5.模型保存路径
        self.model_save_path = HERE / "save_model"
        self.model_predict_path = HERE / "predict_model"


if __name__ == '__main__':
    config = Config()
    print(config.train_data_path)
