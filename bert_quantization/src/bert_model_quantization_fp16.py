"""
BERT fp16 GPU 推理脚本(GPU 部署路线, 与 CPU 动态量化并列)。

- 加载 bert/save_model/bert_classifier_best_model.pt
- 权重转 fp16 + 放 CUDA: 显存减半, 推理快 2~3 倍
- 在测试集上评估(与量化脚本同口径), 并演示单条预测
- 保存 fp16 模型: bert_quantization/save_model/bert_classifier_fp16_model.pt

用法: python bert_quantization/src/bert_model_quantization_fp16.py
"""

import sys
import time
from pathlib import Path

import torch

from bert.src.config import Config

conf = Config()
BERT_SRC = Path(__file__).resolve().parent.parent.parent / "bert" / "src"
sys.path.insert(0, str(BERT_SRC))

from utils import build_dataloader   # noqa: E402
from bert_classifier import TMFBertClassifer  # noqa: E402
from train import model2eval         # noqa: E402


FP16_OUT = Path(__file__).resolve().parent.parent / "save_model" / "bert_classifier_fp16_model.pt"

assert torch.cuda.is_available(), "需要 CUDA GPU"
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1.加载数据
_, _, test_dataloader = build_dataloader()
print(f"测试集已加载: {len(test_dataloader.dataset)} 条")

# 2.加载模型 -> CUDA -> fp16
model = TMFBertClassifer().to("cuda")
model.load_state_dict(torch.load(BERT_SRC.parent / "save_model"  /"bert_classifier_best_model.pt", map_location="cuda"))
model = model.half()     # 权重转 fp16(输入 input_ids 仍是 long, 由 embedding 处理)
model.eval()
print(f"\n已加载最优模型并转 fp16: {BERT_SRC.parent / 'bert_classifier_best_model.pt'}")

# 3.评估(与量化脚本同口径, weighted_f1)
t0 = time.time()
f1, report, confmat = model2eval(test_dataloader, model, "cuda")
elapsed = time.time() - t0
print(f'fp16 模型的 weighted_f1:{f1:.4f} | 测试集评估耗时: {elapsed:.1f}s')
print(f'fp16 模型的分类评估报告:\n{report}')
print(f'fp16 模型的混淆矩阵:\n{confmat}')

# 4.保存 fp16 模型(权重为 fp16, 部署加载后 .float() 可转回)
torch.save(model.state_dict(), FP16_OUT)
print(f"fp16 模型已保存: {FP16_OUT}")

# 4.5 保存后重新加载评估(验证保存的文件可用, 指标应与保存前一致)
loaded = TMFBertClassifer().to("cuda")
loaded.load_state_dict(torch.load(FP16_OUT, map_location="cuda"))
loaded = loaded.half().eval()
f1_loaded, report_loaded, confmat_loaded = model2eval(test_dataloader, loaded, "cuda")
print(f'\n[保存后加载] fp16 模型 weighted_f1:{f1_loaded:.4f}')
print(f'[保存后加载] 混淆矩阵:\n{confmat_loaded}')
assert abs(f1_loaded - f1) < 1e-6, "保存前后指标不一致, 文件可能损坏"
print("保存前后指标一致, fp16 模型文件可用 ✅")

# 5.单条预测演示
sample = "这导演脑子被驴踢了吧"
enc = conf.tokenizer([sample], padding="max_length", max_length=conf.max_len,
                     truncation=True, return_tensors="pt")
with torch.no_grad():
    logits = model(enc["input_ids"].to("cuda"), enc["attention_mask"].to("cuda"))
    pred = int(torch.argmax(logits, dim=-1)[0]) + 1    # 0~3 -> 显示 1~4
print(f'\n单条预测演示: "{sample}" -> {pred}级')
