import joblib
import pandas as pd

from config import Config

config = Config()

# 1. 加载模型
model = joblib.load(config.model_save_path / "rf_model.pkl")
tfidf = joblib.load(config.model_save_path / "tfidf.pkl")

# 2. 获取数据并预测
data = pd.read_csv(config.processed_test_data_path, sep="\t")
X_predict = tfidf.transform(data["words"])
y_pred = model.predict(X_predict)

# 3. 保存结果
config.model_predict_path.mkdir(parents=True, exist_ok=True)
res = pd.DataFrame({"text": data["text"], "category": y_pred})
out = config.model_predict_path / "rf_prediction.csv"
res.to_csv(out, index=False, sep="\t")
print(f"预测完成, 已保存: {out}")
