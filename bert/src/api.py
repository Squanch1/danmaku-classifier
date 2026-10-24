"""
FastAPI 版弹幕分级接口(替代原 Flask 版, 功能不变)。

用法:
  python api.py              # 启动服务(等价于原 Flask app1.run)
  或 uvicorn api:app --host 127.0.0.1 --port 8000
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from predict_fun import predict

# 1.实例化 FastAPI 对象
app = FastAPI()


# 2.定义路由
@app.post('/predict')
async def predict_api(request: Request):
    try:
        # 1.获取请求数据
        data = await request.json()
        # 2.进行模型预测
        res = predict(data)

        # 必须要有返回值
        return res

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# 3.启动服务
if __name__ == '__main__':
    import uvicorn

    # host/port 与原 Flask 版一致
    # fastapi 结合 uvicorn 可实现高并发(协程); 生产可再加 gunicorn 多进程
    uvicorn.run(app, host='127.0.0.1', port=8000)
