# 弹幕内容生态智能分级系统

个人自然语言处理项目：将弹幕分为垃圾／违规、低质、普通、优质四类，覆盖数据构建、分类模型训练、量化与蒸馏实验，以及推理服务。

项目在本地开发，保留了2026年9月的原始实验日志；本仓库于2026年10月整理公开。提交时间对应实际整理发布，不追补过去的 Git 历史。

## 已验证的结果

| 项目 | 结果与口径 |
| --- | --- |
| 原始采集池 | 20万条去重公开弹幕，原始池不随仓库发布 |
| 公开数据快照 | 训练4000条、验证1000条、测试1600条，标签为0至3 |
| 测试集 | 每类400条，其中377条与合成样本记录匹配；不代表真实线上自然分布 |
| 发布 FP16 权重 | 重新加载评估：accuracy=0.845，Weighted-F1=0.8464198617 |
| 原 INT8 实验 | 1600条均衡测试集Weighted-F1=0.8425；模型文件由390.21 MiB降至145.58 MiB |
| 原蒸馏实验 | 某轮BiLSTM学生Weighted-F1=0.8028；保留失败实验日志，不宣称所有后续权重均达到该结果 |

完整复核环境、混淆矩阵及文件校验值见 [reports/reproduced_fp16.json](reports/reproduced_fp16.json)。原日志中的 `macro_f1` 字段有标注不一致问题，原评估代码实际返回Weighted-F1；本次复核分别计算两项。当前测试集各类等量，因此两者数值相同。

## 快速体验：不重新训练

要求Python 3.10。以下命令在克隆后的仓库根执行，脚本内部文件路径均按模块位置解析。

```bash
git clone https://github.com/Squanch1/danmaku-classifier.git
cd danmaku-classifier
uv venv --python 3.10
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Windows虚拟环境解释器为`.venv/Scripts/python.exe`；Linux/macOS为`.venv/bin/python`。下文的`python`指安装好依赖的解释器。PyTorch的CPU/CUDA版本按本机环境选择，本次验证使用的实际版本见报告。

```bash
python -m repro.download_assets
python -m repro.evaluate
python -m pytest -q
python -m uvicorn server:app --app-dir deploy --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/docs`，使用`POST /predict`，请求示例：

```json
{"text":"这里补充一下原理，这个现象由温度变化引起"}
```

响应包含1至4级标签、类别名和模型top-1分数。分数未经概率校准，不视为正确率保证。

最终FP16权重不进入Git历史，通过 [v1.0.0 Release](https://github.com/Squanch1/danmaku-classifier/releases/tag/v1.0.0) 分发。下载脚本核对SHA-256；只下载此权重即可推理与评估，不需要基础BERT权重。

## 重新训练

重新训练需要独立的原始`bert-base-chinese`预训练权重。不能使用已微调的Release权重代替基础模型再宣称从同一起点训练。

```bash
python -m repro.download_pretrained
python -m repro.train --epochs 5 --seed 42
python -m repro.evaluate --checkpoint repro_output/best_model.pt --output repro_output/evaluation.json
```

默认从`google-bert/bert-base-chinese`下载；网络无法访问Hugging Face时，可从ModelScope取得同一模型的`config.json`、分词器与基础权重，放入`bert/bert-base-chinese/`，或用`--pretrained`指定绝对路径。

统一训练入口采用AdamW、学习率3e-5、batch size 32、最大长度128、交叉熵、每25步及每轮末验证，并依据验证集Weighted-F1保存最优模型。随机种子42是本次明确固定的复现配置，旧日志未完整记录历史随机状态；重新训练允许因硬件、软件和随机状态产生波动。**精确核对84.64%的路径是下载发布权重后评估，不承诺任何重新训练都得到同一数字。**

原训练、量化和蒸馏脚本仍保留在对应`src/`目录，用于查看历史实现；它们部分沿用旧数据准备方式和基础模型路径。第一次运行请使用`repro/`统一入口。

## 目录

| 目录 | 内容 |
| --- | --- |
| `repro/` | 本次补充的统一下载、训练、评估入口 |
| `bert/src/` | 原BERT训练、评估与数据处理代码 |
| `bert/processed_data/` | 本次固定的数据快照，0至3标签 |
| `bert_distll/` | 原BiLSTM与蒸馏实验、成功和失败日志 |
| `bert_quantization/` | FP16及INT8动态量化实验 |
| `fasttext_train/`、`RamdomForest/`、`Roberta/` | 基线与探索代码，不同历史数据口径不得直接计算提升幅度 |
| `data/scripts/` | 原采集、标注和候选数据挖掘脚本，运行可能调用外部平台或付费模型 |
| `deploy/` | FastAPI、动态批处理、启动预热、Docker配置 |
| `reports/` | 本次复核、原文件来源清单及公开整理说明 |

## 数据与使用边界

公开快照只包含文本与标签，不包含用户账号等采集元数据。数据来自公开弹幕及合成补充，标注由大模型辅助完成，未建立完整人工金标准；该模型是学习实验，不应作为无人复核的实际内容处罚依据。数据、基础模型及依赖的使用还须遵循各自来源条款，本仓库不将第三方内容重新声明为作者独占所有。

原训练README和部署文档作为历史笔记保留，其中自然分布数据量、旧环境和压测数字不代表本次公开版本的验证结果。部署文档记录的吞吐未在本次复测，不纳入上述结果表。

## 个人开发记录与本次整理

项目工作围绕采集与分级标注、多模型实验、模型压缩、蒸馏和服务化展开。原实验源码和日志保留在仓库，`reports/source_manifest.json`记录导出时文件校验值。公开整理增加了统一入口、检查、文档与发布配置；它们与原历史实验分别记录，不补造开发过程。

参见 [公开整理说明](reports/PUBLICATION.md)。
