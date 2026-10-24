from pathlib import Path

import jieba
import pandas as pd

from config import Config

config = Config()


# todo 1.分词预处理函数
def cut_sentence(s):
    return " ".join(jieba.lcut(str(s)))


# todo 2.数据处理函数
def data_processing(raw_path, processed_path):
    raw_path = Path(raw_path)
    # 1.读取文件(jsonl: {"text":..., "label":...}; 兼容旧的 tab 分隔 txt)
    if raw_path.suffix == ".jsonl":
        raw_data = pd.read_json(raw_path, lines=True)[["text", "label"]]
    else:
        raw_data = pd.read_csv(raw_path, sep="\t", names=["text", "label"])
    # 2.处理数据
    raw_data["category"] = raw_data["label"]
    raw_data["words"] = raw_data["text"].apply(cut_sentence)
    # 3.保存文件
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    raw_data.to_csv(processed_path, sep="\t", index=False)
    print(f"文件保存成功!,路径为{processed_path}")


# todo 3.主处理函数
def main():
    data_processing(config.train_data_path, config.processed_train_data_path)
    data_processing(config.test_data_path, config.processed_test_data_path)
    data_processing(config.dev_data_path, config.processed_dev_data_path)


if __name__ == '__main__':
    main()
