import requests
import time
url = "http://192.168.11.88:8000/predict"


data = {'text': "中国人民公安大学2012年硕士研究生目录及书目"}
start_time = time.time()
res = requests.post(url, json=data)
use_time = time.time() - start_time

# 打印预测结果和耗时
print(f'predict result:{res.json()}')
print(f'use_time:{use_time}s')








