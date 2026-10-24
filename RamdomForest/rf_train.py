import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV

from config import Config

# 1.获取数据 + 数据预处理
config = Config()
# 1.1 训练集数据
train_data = pd.read_csv(config.processed_train_data_path, sep="\t")
train_words = train_data["words"]
print(train_words.shape)
y_train = train_data["category"]
# 1.2 验证集数据
dev_data = pd.read_csv(config.processed_dev_data_path, sep="\t")
dev_words = dev_data["words"]
y_dev = dev_data["category"]
# 1.3 测试集数据
test_data = pd.read_csv(config.processed_test_data_path, sep="\t")
test_words = test_data["words"]
y_test = test_data["category"]

# 2. 特征工程
# 2.1 读取停用词(文件不存在则用空表)
stopwords_path = Path(config.stopwords_path)
stopwords = stopwords_path.read_text(encoding="utf-8").split() if stopwords_path.exists() else []
# 2.2 构建TfidfVectorizer实例化对象
tfdif = TfidfVectorizer(stop_words=stopwords)
# 2.3 对训练集数据向量化
X_train = tfdif.fit_transform(train_words)
X_dev = tfdif.transform(dev_words)
X_test = tfdif.transform(test_words)

# 3.模型训练
# 3.1 选择随机森林模型
model = RandomForestClassifier(random_state=42)
# 3.2 创建交叉验证,网格搜索的对象(class_weight 处理 1 级样本稀缺)
# 注意: 必须用 f1_macro 作为网格评分——默认 accuracy 会选出"忽略稀有类"的模型
grid_cv = GridSearchCV(model, param_grid={
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [10, 15, 20],
    "class_weight": ["balanced", None],
    "n_jobs": [-1]  # 使用cpu核心数
}, scoring="f1_macro")
grid_cv.fit(X_train, y_train)

# 4.模型评估
best_model = grid_cv.best_estimator_
print(f"最佳参数:{grid_cv.best_params_}")
y_pred = best_model.predict(X_test)
# 模型评估
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
mae = float(np.mean(np.abs(np.array(y_test, dtype=int) - np.array(y_pred, dtype=int))))
print(f"accuracy: {acc:.3f} | weighted-F1: {f1:.3f} | macro-F1: {f1_macro:.3f} | MAE: {mae:.2f}")
print("混淆矩阵(行=真实, 列=预测):")
print(confusion_matrix(y_test, y_pred))

# 5.模型保存
config.model_save_path.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, config.model_save_path / "rf_model.pkl")
joblib.dump(tfdif, config.model_save_path / "tfidf.pkl")
(config.model_save_path / "best_params.json").write_text(
    json.dumps(grid_cv.best_params_, ensure_ascii=False), encoding="utf-8")
print(f"词汇表大小: {len(tfdif.vocabulary_)}")
print(f"模型已保存到: {config.model_save_path}")
