# 生产部署镜像 (CUDA GPU 推理)
# 基础镜像可按环境调整, 例如 pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# 先装依赖再拷代码, 利用镜像层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py inference.py server.py ./
COPY model/ ./model/

EXPOSE 8000

# 单 worker: 模型在进程内常驻, 多 worker 会重复占用显存; 扩并发用多副本/多容器
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
