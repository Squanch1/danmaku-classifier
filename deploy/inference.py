"""
推理模块: 模型加载(FP16 GPU / FP32 CPU 自动切换) + 启动预热 + 单条/批处理预测。
等价于本地 bert/src 的 predict_fun.py + bert_classifier.py, 部署自包含。
"""
import queue
import threading
import time

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

from config import Config

conf = Config()


class TMFBertClassifier(nn.Module):
    """模型结构与训练时 bert_classifier.py 完全一致 (参数名必须一致才能加载 state_dict)。"""

    def __init__(self):
        super().__init__()
        # 只用 config.json 构建架构(随机初始化), 随后 load_state_dict 覆盖全部权重,
        # 因此不需要 411MB 的预训练基础权重
        self.bert = BertModel(BertConfig.from_pretrained(conf.bert_path))
        self.fc = nn.Linear(self.bert.config.hidden_size, conf.class_num)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(output.pooler_output)
        return logits


class Predictor:
    """进程内常驻模型: 加载一次 + 预热一次, 每个请求复用。"""

    def __init__(self, config=None):
        self.conf = config or Config()
        self.tokenizer = BertTokenizer.from_pretrained(self.conf.bert_path)
        self.model = self._load_model()
        self.warmup()

    def _load_model(self):
        model = TMFBertClassifier().to(self.conf.device)
        state_dict = torch.load(str(self.conf.model_path), map_location=self.conf.device)
        model.load_state_dict(state_dict)
        # fp16 权重: GPU 上转 half 加速; CPU 上 load_state_dict 已隐式转 float32, 保持 float 运行
        if self.conf.device.type == "cuda":
            model = model.half()
        model.eval()
        return model

    def warmup(self):
        """启动预热: 提前触发 CUDA kernel 初始化, 避免第一个请求冷启动慢/失败。"""
        with torch.no_grad():
            self.predict("预热")

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """批量预测: 按 batch 内最长长度 padding(弹幕平均 ~13 字, 远快于固定 128)。"""
        if not texts:
            return []
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.conf.max_len,
            padding="longest",     # 与训练一致由 attention_mask 屏蔽 padding, 结果不变
            truncation=True,
        )
        with torch.no_grad():
            input_ids = enc.input_ids.to(self.conf.device)
            attention_mask = enc.attention_mask.to(self.conf.device)
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1)

        results = []
        for i in range(len(texts)):
            idx = int(pred_idx[i].item())
            results.append({
                "label": idx + 1,                        # 1~4 级
                "pred_class": self.conf.class_list[idx],
                "prob": round(float(probs[i, idx].item()), 4),   # top-1 置信度
            })
        return results


class _Slot:
    """批处理请求的回执: 调用方阻塞等待 worker 线程填充结果。"""

    def __init__(self):
        self.event = threading.Event()
        self.result = None


class BatchedPredictor:
    """动态批处理: 请求进队列, 攒够 batch_size 或超时 max_wait 后一次性前向。
    单 worker 即可支撑数百 QPS (batch=100 时实测 ~667 QPS)。"""

    def __init__(self, predictor: Predictor, batch_size: int = None, max_wait: float = None):
        self.predictor = predictor
        self.batch_size = batch_size or conf.batch_size
        self.max_wait = max_wait if max_wait is not None else conf.max_wait
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def predict(self, text: str) -> dict:
        slot = _Slot()
        self._queue.put((text, slot))
        if not slot.event.wait(timeout=30):      # 兜底超时, 避免极端情况永久阻塞
            raise TimeoutError("批处理推理超时")
        return slot.result

    def _run(self):
        while True:
            batch = [self._queue.get()]
            deadline = time.time() + self.max_wait
            # 攒批: 尽量攒满, 但最多等 max_wait 秒
            while len(batch) < self.batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                results = self.predictor.predict_batch([item[0] for item in batch])
            except Exception as e:                # 单个批出错不影响其他请求
                results = [{"error": str(e)}] * len(batch)
            for (_, slot), r in zip(batch, results):
                slot.result = r
                slot.event.set()


if __name__ == "__main__":
    p = Predictor()
    for t in ["哈哈哈哈哈哈", "前方高能预警", "加V看片:xxxx", "科普:视频里讲错了,正确的是……"]:
        print(t, "->", p.predict(t))
