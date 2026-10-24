"""
性能基准: 拆解 tokenizer / 串行 / 批处理 的耗时, 决定是否需要批处理或多 worker。
用法: python bench.py
"""
import threading
import time
import threading

import torch

from config import Config
from inference import BatchedPredictor, Predictor

conf = Config()
p = Predictor()

samples = [
    "前方高能预警,这波操作太秀了",
    "哈哈哈哈哈哈",
    "加V看片:xxxx",
    "up主说的对,这里其实是致敬了XX电影",
    "科普:视频里第二定律讲错了,正确的是……",
] * 20  # 100 条

# 1) tokenizer 只耗时
t0 = time.time()
for s in samples:
    p.tokenizer(s, return_tensors="pt", max_length=conf.max_len,
                padding="max_length", truncation=True)
t_tok = time.time() - t0
print(f"[1] tokenizer 100 条: {t_tok:.2f}s  -> 单条 {(t_tok/100)*1000:.1f}ms")

# 2) 串行完整预测
t0 = time.time()
for s in samples:
    p.predict(s)
t_ser = time.time() - t0
print(f"[2] 串行预测 100 条: {t_ser:.2f}s -> {100/t_ser:.0f} QPS (单条 {(t_ser/100)*1000:.1f}ms)")

# 3) 批处理前向(100 条一次 batch)
enc = p.tokenizer(samples, return_tensors="pt", max_length=conf.max_len,
                  padding="max_length", truncation=True)
with torch.no_grad():
    t0 = time.time()
    _ = p.model(enc.input_ids.to(conf.device), enc.attention_mask.to(conf.device))
    t_batch = time.time() - t0
print(f"[3] batch=100 单次前向: {t_batch*1000:.1f}ms -> 理论 {100/(t_batch):.0f} QPS")

# 4) 批处理 + 动态长度(按真实长度 padding, 弹幕很短, 更快)
real = [len(p.tokenizer.tokenize(s)) for s in samples]
print(f"[4] 样本平均真实长度: {sum(real)/len(real):.0f} 字 (max_len=128 多数是 padding)")

# 5) BatchedPredictor 并发吞吐(模拟服务端动态批处理)
bp = BatchedPredictor(p)
N_THREADS, PER_THREAD = 8, 50   # 8 线程 x 50 次 = 400 请求
total = N_THREADS * PER_THREAD
errs = [0]
barrier = threading.Barrier(N_THREADS)
t0 = time.time()
def _worker():
    barrier.wait()
    for i in range(PER_THREAD):
        r = bp.predict(samples[i % len(samples)])
        if "error" in r:
            errs[0] += 1
threads = [threading.Thread(target=_worker) for _ in range(N_THREADS)]
for t in threads: t.start()
for t in threads: t.join()
dt = time.time() - t0
print(f"[5] BatchedPredictor {total} 请求: {dt:.2f}s -> {total/dt:.0f} QPS, 错误 {errs[0]}")
print(f"    平均每条延迟(含排队): {(dt/total)*1000:.1f}ms")
