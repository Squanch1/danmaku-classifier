from pathlib import Path

import fasttext

from config import Config
from processing_data import cut_sentence_by_char

config = Config()
# 1.加载模型(路径锚定 fasttext_train/save_models)
model = fasttext.load_model(str(config.save_model_path / "fasttext_char_auto.bin"))


def predict(data: dict):
    words = cut_sentence_by_char(data["text"])
    res = model.predict(words)
    data["predict"] = res[0][0][9:]
    return data


if __name__ == "__main__":
    data = {"text": "体验2D巅峰倚天屠龙记十大创新概览"}
    print(predict(data))
