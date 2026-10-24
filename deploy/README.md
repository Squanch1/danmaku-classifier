# 弹幕质量分级 · 线上部署

模型: **BERT-base-Chinese fine-tuned (fp16)**, 4 级单标签分类:
1级 垃圾/违规 · 2级 低质 · 3级 普通 · 4级 优质。

部署目录自包含, 不依赖训练代码(`bert/src` 等)。模型权重与 tokenizer 均在 `model/` 下。

## 目录结构

```
deploy/
├── server.py            # FastAPI 入口: /predict /health
├── inference.py         # 模型加载 + 预热 + 预测 (Predictor)
├── config.py            # 路径/设备/类别/超参配置
├── requirements.txt     # 推理依赖
├── Dockerfile
├── docker-compose.yml   # GPU 部署
├── test_api.py          # 本地联调
├── bench.py             # 性能基准(串行/批处理/动态批)
├── locustfile.py        # 压测
└── model/
    ├── bert_classifier_fp16_model.pt   # fp16 权重 (204MB)
    ├── class.txt                      # 类别表
    └── bert-base-chinese/             # tokenizer 小文件 (无基础权重)
```

## 本地运行 (推荐训练环境 pytorch_gpu)

```bash
conda activate pytorch_gpu
cd deploy
pip install -r requirements.txt          # 训练环境已具备 torch/transformers 可跳过
uvicorn server:app --host 0.0.0.0 --port 8000
```

启动日志会打印模型加载耗时; 首次请求前已自动预热。

## 联调 & 压测

```bash
python test_api.py            # 健康检查 + 预测 + 参数校验

python bench.py               # 拆解 tokenizer/串行/批处理/动态批吞吐

pip install locust
locust -f locustfile.py --host http://127.0.0.1:8000   # 打开 http://localhost:8089
```

## 性能实测(训练环境 pytorch_gpu, 2080Ti 级 GPU, fp16)

| 模式 | 结果 |
|---|---|
| 串行预测 | ~97 QPS(10ms/条) |
| 单次 batch=100 前向 | 150~160ms → 理论 ~600 QPS |
| **动态批处理(服务默认)** | **255 QPS(进程内)** / **197 QPS(HTTP 端到端)**, 0 错误 |
| HTTP 延迟 | p50 87ms · p95 141ms · p99 215ms(20 并发) |

动态批处理参数在 `config.py`: `batch_size=32` 攒批上限, `max_wait=0.01s` 最长攒批等待。
调吞吐增大 `max_wait`(攒更大批); 调延迟减小它。默认即可满足 >100 QPS。

## Docker + GPU 部署

```bash
cd deploy
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

- 默认 **单 worker**: 模型在进程内常驻, 多 worker 会重复占显存。
- 扩并发建议: 加容器副本 + 前置负载均衡; 需要极高吞吐时再引入 Triton/动态 batch。

## 接口约定

```
GET  /health                     -> {"status":"ok","model_loaded":true}
POST /predict  {"text":"..."}    -> {"code":0,"data":{
                                        "text":"...",
                                        "label":3,          # 1~4 级
                                        "pred_class":"3级",
                                        "prob":0.91,        # top-1 置信度
                                        "latency_ms":5.2
                                     },"msg":"ok"}
```

校验规则: `text` 必填、非空、≤100 字; 违返回 400。
