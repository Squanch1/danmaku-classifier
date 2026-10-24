from flask import Flask, request
from predict_fun import predict

# 1.创建Flask对象
app = Flask(__name__)


# 2.定义路由(请求路径,请求参数)
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        res = predict(data)
        return res
    except Exception as e:
        return {"error": str(e), "status code": 400}

