from predict_fun import predict
from flask import Flask, request  # request: 获取请求数据

# 1.实例化Flask对象
app1 = Flask(__name__)

# 2.定义装饰的路由
@app1.route('/predict', methods=['POST'])
def predict_api():
    try:
        # 1.获取请求数据
        data = request.get_json()
        # 2.进行模型预测
        res = predict(data)
        # 必须要有返回值
        return res

    except Exception as e:
        return {"error": str(e)}, 400
# 3.启动服务
if __name__ == '__main__':
    # flask   结合gunicorn 可以实现多进程
    # fastapi 结合uvicorn 可以实现多线程（协程）
    app1.run(host='192.168.11.110', port=8000, debug=False)

