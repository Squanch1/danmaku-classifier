from flask import Flask,request

# 创建Flask对象:
app1 = Flask(__name__) 

# 定义装饰器的路由
@app1.route("/predict_rf",method=["POST"])
def predict():
    try:
        pass
    except:
        pass
