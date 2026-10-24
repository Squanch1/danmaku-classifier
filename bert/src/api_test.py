import requests
import time
url = "http://127.0.0.1:8000/predict"


data = {'text': "祖国繁荣昌盛"}
start_time = time.time()
res = requests.post(url, json=data)
use_time = time.time() - start_time

# 打印预测结果和耗时
print(f'predict result:{res.json()}')
print(f'use_time:{use_time}s')








