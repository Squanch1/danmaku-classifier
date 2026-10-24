import time
from datetime import datetime
from pathlib import Path

import fasttext
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from processing_data import cut_sentence_by_char
from config import Config
import pandas as pd

config = Config()
# 1.模型训练(autotune 调参)
model = fasttext.train_supervised(input=str(config.processed_data_path / "train_char.csv"),
                                  autotuneValidationFile=str(config.processed_data_path / "dev_char.csv"),
                                  autotuneDuration=30,
                                  seed=1)
# 2.模型保存
config.save_model_path.mkdir(parents=True, exist_ok=True)
model.save_model(str(config.save_model_path / "fasttext_char_auto.bin"))
# 3.模型预测
print(model.predict(cut_sentence_by_char("体验2D巅峰倚天屠龙记十大创新概览")))
# 4.模型评估(在测试集上,用真实标签算指标)

test_data = pd.read_csv(config.processed_data_path / "test_char.csv", sep="\t")
y_true = [int(l.replace("__label__", "")) for l in test_data["label"]]
y_pred = [int(model.predict(t, k=1)[0][0][9:]) for t in test_data["text"]]
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
acc = accuracy_score(y_true, y_pred)
print(f"accuracy: {acc:.3f} | macro-F1: {f1:.3f}")

# 5.训练完成, 输出日志到 fasttext_train/logs/
log_dir = config.save_model_path.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"char_auto_{datetime.now():%Y%m%d_%H%M}.log"
with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"=== fastText char_auto 训练日志 ===\n")
    f.write(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    f.write(f"模型: fasttext_char_auto.bin(字符切分 + autotune)\n")
    f.write(f"测试集: {len(y_true)} 条(均衡 1:1:1:1)\n")
    f.write(f"accuracy: {acc:.3f} | macro-F1: {f1:.3f}\n")
print(f"训练日志已保存: {log_path}")


