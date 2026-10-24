from pathlib import Path

import jieba
import pandas as pd

from config import Config

config = Config()


# 按词划分
def cut_sentence_by_word(sentence):
    return " ".join(jieba.cut(str(sentence)))


# 按字符划分
def cut_sentence_by_char(sentence):
    return " ".join([char.strip() for char in list(str(sentence)) if char.strip()])


def create_label(label):
    return "__label__" + str(label)


def process_data(raw_data_path, processed_data_path, is_char_segmention=False,
                 oversample=None):
    """oversample: {label: 倍数} —— 对稀有类复制样本, 让 fastText 损失重视它们。
    例: oversample={'1': 20} 表示 1 级样本重复 20 倍。"""
    raw_data_path = Path(raw_data_path)
    # jsonl: {"text":..., "label":...}; 兼容旧 tab 分隔 txt
    if raw_data_path.suffix == ".jsonl":
        raw_data = pd.read_json(raw_data_path, lines=True)[["text", "label"]]
    else:
        raw_data = pd.read_csv(raw_data_path, sep='\t', names=["text", "label"])
    if is_char_segmention:
        series_text = raw_data["text"].apply(cut_sentence_by_char)
    else:
        series_text = raw_data["text"].apply(cut_sentence_by_word)
    series_label = raw_data["label"].apply(create_label)
    processed_train_data = pd.DataFrame({"label": series_label, "text": series_text})
    if oversample:
        extra = []
        for label, times in oversample.items():
            part = processed_train_data[processed_train_data["label"] == f"__label__{label}"]
            for _ in range(times - 1):
                extra.append(part)
        if extra:
            processed_train_data = pd.concat([processed_train_data, *extra])
            processed_train_data = processed_train_data.sample(frac=1, random_state=42)
    processed_train_data.to_csv(processed_data_path, index=False, sep="\t")


def main():
    processed_dir = Path(config.processed_data_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    # 数据已 1:1:1:1 均衡, 无需过采样
    # 按字符分词
    process_data(config.train_data_path, processed_dir / "train_char.csv", is_char_segmention=True)
    process_data(config.test_data_path, processed_dir / "test_char.csv", is_char_segmention=True)
    process_data(config.val_data_path, processed_dir / "dev_char.csv", is_char_segmention=True)

    # 按词分词
    process_data(config.train_data_path, processed_dir / "train_word.csv", is_char_segmention=False)
    process_data(config.test_data_path, processed_dir / "test_word.csv", is_char_segmention=False)
    process_data(config.val_data_path, processed_dir / "dev_word.csv", is_char_segmention=False)


if __name__ == '__main__':
    main()
