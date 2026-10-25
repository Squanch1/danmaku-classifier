# 弹幕质量分级 · 开发测试手册

> 时间说明：本项目于2024年9月（大二上学期）在本地开发；本文为保留的实验笔记，其中2026年日期是原文件时间标记，与项目最初开发时间分开记录。2026年10月的公开整理及复现说明见[公开整理说明](../reports/PUBLICATION.md)。

> 面向开发/测试人员。部署目录 `deploy/` 自包含,不依赖训练代码(`bert/src` 等)。
> 模型:BERT-base-Chinese fine-tuned(fp16),4 级单标签分类。

---

## 1. 任务与模型概览

| 项 | 说明 |
|---|---|
| 任务 | 弹幕质量单标签 4 分类 |
| 类别 | 1级 垃圾/违规 · 2级 低质 · 3级 普通 · 4级 优质 |
| 模型 | BERT-base-Chinese(fine-tuned),权重 fp16 保存 |
| 输入 | 弹幕文本,≤100 字,内部 pad 到 max_len=128 |
| 技术栈 | PyTorch + Transformers + FastAPI + uvicorn(动态批处理) |
| 测试集精度 | macro-F1 ≈ 0.85(fp32/fp16 一致) |

---

## 2. 目录结构

```
deploy/
├── server.py            # FastAPI 入口: /predict /health
├── inference.py         # Predictor(加载/预热/预测) + BatchedPredictor(动态批处理)
├── config.py            # 路径/设备/批处理参数配置
├── test_api.py          # HTTP 联调脚本
├── bench.py             # 性能基准
├── locustfile.py        # 压测(需装 locust)
├── requirements.txt     # 推理依赖
├── Dockerfile / docker-compose.yml   # 生产部署用(开发可忽略)
├── MANUAL.md            # 本文档
├── README.md            # 快速上手
└── model/
    ├── bert_classifier_fp16_model.pt  # fp16 权重 (204MB)
    ├── class.txt                      # 类别表
    └── bert-base-chinese/             # tokenizer 小文件
```

---

## 3. 环境准备

训练/开发环境是 conda 的 `pytorch_gpu`(已装好 torch 2.13.0+cu126、transformers 5.15.1)。

```bash
conda activate pytorch_gpu
```

如果要在全新环境装依赖:

```bash
pip install -r requirements.txt
```

---

## 4. 快速开始(启动服务)

> ⚠️ **必须从 `deploy/` 目录启动**,否则 `uvicorn server:app` 找不到 `server` 模块,直接报错退出。

```bash
conda activate pytorch_gpu
cd E:\code\danmaku_classifier\deploy
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

启动日志会打印 `模型加载完成, 设备=cuda, 耗时 Xs`(已自动预热)。

- 本机访问:http://127.0.0.1:8000
- **Swagger 调试页(推荐):http://127.0.0.1:8000/docs** —— 可视化点按钮发请求
- 健康检查:http://127.0.0.1:8000/health → `{"status":"ok","model_loaded":true}`
- 停止:终端按 `Ctrl+C`

---

## 5. 接口文档

### GET /health

健康检查(供负载均衡/探针用)。

```json
{"status": "ok", "model_loaded": true}
```

### POST /predict

请求体:

```json
{"text": "前方高能预警"}
```

响应:

```json
{
  "code": 0,
  "data": {
    "text": "前方高能预警",
    "label": 3,              // 1~4 级(数值)
    "pred_class": "3级",     // 等级名
    "prob": 0.9307,          // top-1 置信度
    "latency_ms": 38.5       // 服务端推理耗时(含排队)
  },
  "msg": "ok"
}
```

### 校验规则与错误码

| 场景 | 状态码 | 说明 |
|---|---|---|
| `text` 为空 / 超过 200 字符 | 422 | pydantic 参数校验(`min_length=1, max_length=200`) |
| `text` 去掉空白后为空 | 400 | detail: `text 为空` |
| 超过 100 字 | 400 | detail: `弹幕长度超过 100 字` |
| 推理异常 | 500 | detail: 具体错误信息 |
| 模型未加载完成 | 503 | detail: `模型尚未加载完成` |

---

## 6. 开发联调

### 方式一:Swagger(最省事)

打开 http://127.0.0.1:8000/docs → 展开 `/predict` → `Try it out` → 填 `{"text":"66666"}` → `Execute`。

### 方式二:test_api.py(批量测样例 + 校验)

服务启动后,**另开一个终端**运行:

```bash
cd E:\code\danmaku_classifier\deploy
python test_api.py
```

输出各样例的预测结果、置信度和耗时,并验证空文本被拦截。

### 方式三:直接跑模型(不走 HTTP)

```bash
cd E:\code\danmaku_classifier\deploy
python inference.py
```

### 方式四:curl(注意中文编码)

Windows 的 Git Bash/控制台默认 GBK,`-d '{"text":"中文"}'` 容易乱码报 `error parsing the body`。建议:
- 把 JSON 写入文件再 `-d @body.json`;或
- 直接用 Python `requests`(见 test_api.py);或
- 用 Swagger。

---

## 7. 性能与调优

### 基准测试

```bash
cd E:\code\danmaku_classifier\deploy
python bench.py
```

输出 tokenizer / 串行 / 单次大 batch / 动态批处理 的耗时与 QPS。

### 实测数据(2080Ti 级 GPU, fp16, 2080Ti 级 GPU)

| 模式 | 结果 |
|---|---|
| 串行预测 | ~97 QPS(10ms/条) |
| 单次 batch=100 前向 | ~160ms → 理论 ~600 QPS |
| 动态批处理(服务默认) | ~255 QPS(进程内) / ~197 QPS(HTTP 端到端),0 错误 |
| HTTP 延迟 | p50 87ms · p95 141ms · p99 215ms(20 并发) |

### 调优旋钮(`config.py`)

```python
self.batch_size = 32   # 攒批上限
self.max_wait = 0.01   # 最长攒批等待(秒)
```

- 想**更高吞吐**:增大 `max_wait`(攒更大的批),例如 0.03
- 想**更低延迟**:减小 `max_wait`,例如 0.005
- 默认即满足 >100 QPS;当前瓶颈在排队而非模型本身

---

## 8. 模型与切换

### 当前模型

`model/bert_classifier_fp16_model.pt`(fp16,204MB)。加载逻辑在 `inference.py::_load_model`:

```python
model = TMFBertClassifier().to(device)          # 用 BertConfig 纯构建架构
model.load_state_dict(torch.load(path, map_location=device))
if device.type == "cuda":
    model = model.half()    # GPU 用 fp16 加速
# CPU 上 load_state_dict 已隐式转 float32, 保持 float 运行
model.eval()
```

> 关键:架构用 `BertConfig` 从 `model/bert-base-chinese/config.json` 构建,**不需要** 411MB 预训练权重(加载的 state_dict 会覆盖全部参数)。

### 切换其他模型(训练产物)

| 想用的模型 | 文件 | 改哪里 |
|---|---|---|
| 原版 BERT fp32 | `bert/save_model/bert_classifier_best_model.pt` | `config.py` 的 `model_path` |
| BERT qint8 | `bert_quantization/save_model/bert_classifier_qint8_model.pt` | `model_path` + 用 `torch.load(..., weights_only=False)` 整模型加载 |
| 蒸馏 BiLSTM 学生 | `bert_distll/save_model/bilstm_distill_best_model.pt` | 替换 `inference.py` 的模型类为 `BILSTMClassifier`(见 `bert_distll/src/student_model.py`) |

---

## 9. 常见问题 FAQ

### Q1:启动直接报错退出(exit code 3)

**原因**:不是从 `deploy/` 目录启动,`uvicorn server:app` 找不到 `server` 模块。

**解决**:先 `cd E:\code\danmaku_classifier\deploy` 再启动。

### Q2:端口被占用

```bash
netstat -ano | findstr :8000   # 找占用 PID
taskkill /PID <PID> /F
```

### Q3:控制台中文乱码

Windows 控制台默认 GBK。运行脚本前设:

```bash
set PYTHONIOENCODING=utf-8
```

或直接用 Swagger / Python requests(接口返回的 UTF-8 JSON 本身是正确的)。

### Q4:`{"detail":"There was an error parsing the body"}`

curl 在 Git Bash 里传中文 JSON 的编码问题,不是服务问题。改用文件、Python 或 Swagger(见第 6 节方式四)。

### Q5:模型边界行为观察(业务相关)

- `"66666"` → **2级**(置信度 0.98)。README 标注规范里 "666=3级",但训练模型把它判成低质灌水 —— 属于模型质量问题,不是部署问题。上线前建议在测试集核查"数字刷屏"这类样本,必要时微调或加阈值策略。
- 1 级(有害类)是安全关键类,线上建议单独监控该类命中率。

### Q6:显存/资源

fp16 模型约占用 ~400MB 显存(GPU 空闲 6GB 足够)。单 worker 即可支撑 ~200 QPS;不要多开 worker(每 worker 重复占显存且批处理会相互竞争)。

---

## 10. 生产部署备忘(开发完成后)

### 裸机(Linux)

```bash
# 运行(从 deploy/ 目录)
conda activate pytorch_gpu && uvicorn server:app --host 0.0.0.0 --port 8000
# 守护用 systemd 或 supervisor(开机自启/崩溃重启/日志落盘)
```

### Docker + GPU

```bash
cd deploy
docker compose up -d --build
```

前提:宿主机已装 nvidia-container-toolkit。

---

*生成日期:2026-09-02 · 与 `deploy/` 代码同步*
