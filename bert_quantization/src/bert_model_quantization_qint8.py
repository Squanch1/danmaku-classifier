"""
BERT 动态量化脚本(适配 bert/src 新配置)。

- 复用 bert/src 的 Config/utils/bert_classifier/train(processed_data jsonl, 0~3 标签, 均衡数据)
- 加载 bert/save_model/bert_classifier_best_model.pt(训练好的最优模型)
- 对 Linear 层做动态量化(qint8), CPU 上评估后保存
- 量化模型保存: bert_quantization/save_model/bert_classifier_quantization_model.pt

qint8 只能在cpu上运行
"""

import sys
from pathlib import Path

import torch
from torch.quantization import quantize_dynamic

# 复用 bert/src 的模块(优先于本目录下的旧版同名模块)
BERT_SRC = Path(__file__).resolve().parent.parent.parent / "bert" / "src"
sys.path.insert(0, str(BERT_SRC))

from utils import build_dataloader  # noqa: E402
from bert_classifier import TMFBertClassifer  # noqa: E402
from train import model2eval  # noqa: E402

QUANT_OUT = Path(__file__).resolve().parent.parent / "save_model" / "bert_classifier_qint8_model.pt"

# 1.加载数据(测试集 1600 条, 均衡)
_, _, test_dataloader = build_dataloader()
print(f"测试集已加载: {len(test_dataloader.dataset)} 条")

# 2.加载训练好的最优模型(CPU)
model = TMFBertClassifer().to("cpu")
model.load_state_dict(torch.load(BERT_SRC.parent / "save_model" / "bert_classifier_best_model.pt", map_location="cpu"))
model.eval()
print(f"已加载最优模型: {BERT_SRC.parent/  'save_model' / 'bert_classifier_best_model.pt'}")

# 3.模型量化(仅量化 Linear 层, 动态量化)
quantize_model = quantize_dynamic(
    model,  # 需要量化的模型
    {torch.nn.Linear},  # 需要量化的层
    dtype=torch.qint8  # 量化类型
)
print("\n动态量化完成(qint8, Linear 层)")

# 4.模型评估(CPU 上, 返回 model2eval 的三个值)
f1, report, confmat = model2eval(test_dataloader, quantize_model, "cpu")
print(f'量化模型的 weighted_f1:{f1:.4f}')
print(f'量化模型的分类评估报告:\n{report}')
print(f'量化模型的混淆矩阵:\n{confmat}')

# 5.模型保存(保存整个模型对象: 参数+架构, 只能在 CPU 上运行)
torch.save(quantize_model, QUANT_OUT)
print(f"量化模型已保存: {QUANT_OUT}")

# 6.保存后重新加载评估(验证保存的文件确实可用, 指标应与保存前一致)
loaded_model = torch.load(QUANT_OUT, map_location="cpu", weights_only=False)
loaded_model.eval()
f1_loaded, report_loaded, confmat_loaded = model2eval(test_dataloader, loaded_model, "cpu")
print(f'\n[保存后加载] 量化模型 weighted_f1:{f1_loaded:.4f}')
print(f'[保存后加载] 混淆矩阵:\n{confmat_loaded}')
assert abs(f1_loaded - f1) < 1e-6, "保存前后指标不一致, 文件可能损坏"
print("保存前后指标一致, 量化模型文件可用 ✅")
