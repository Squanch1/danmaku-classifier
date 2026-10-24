"""本地联调: 先启动服务 (uvicorn server:app --port 8000), 再运行本脚本。"""
import time

import requests

BASE = "http://127.0.0.1:8000"


def test_health():
    r = requests.get(f"{BASE}/health")
    print("health:", r.json())


def test_predict():
    samples = [
        "哈哈哈哈哈哈",
        "前方高能预警",
        "加V看片:xxxxxx",
        "科普:视频里第二定律讲错了,正确的是……",
        "66666",
        "up主说的对,这里其实是致敬了XX电影",
    ]
    for s in samples:
        t0 = time.time()
        r = requests.post(f"{BASE}/predict", json={"text": s})
        print(f"{s!r} -> {r.json()} | {(time.time()-t0)*1000:.0f}ms")


def test_validation():
    r = requests.post(f"{BASE}/predict", json={"text": ""})
    print("空文本 ->", r.status_code, r.json())


if __name__ == "__main__":
    test_health()
    test_predict()
    test_validation()
