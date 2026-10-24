"""
通用模型评估脚本: 传入模型路径, 在测试集上评估, 保存日志到 log/。

兼容两种保存格式:
  - int8 动态量化模型: 整对象保存(torch.save(model)) → torch.load 直接加载
  - fp16 模型: state_dict 保存 → 重建模型 + load_state_dict

用法:
  python bert_quantization/src/evaluate_model.py --model ../save_model/bert_classifier_quantization_model.pt
  python bert_quantization/src/evaluate_model.py --model ../save_model/bert_classifier_fp16_model.pt --device auto

日志输出: bert_quantization/log/<模型名>_eval_<时间戳>.txt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

BERT_SRC = Path(__file__).resolve().parent.parent.parent / "bert" / "src"
sys.path.insert(0, str(BERT_SRC))

from utils import build_dataloader   # noqa: E402
from bert_classifier import TMFBertClassifer  # noqa: E402
from train import model2eval         # noqa: E402

LOG_DIR = Path(__file__).resolve().parent.parent / "log"


def load_model(path, device):
    """加载模型, 自动区分 int8(整对象)与 fp16(state_dict)两种格式。"""
    # weights_only=False: 整对象保存的量化模型含自定义类, PyTorch 2.6+ 默认拦截
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):                       # state_dict 格式(fp16)
        model = TMFBertClassifer().to(device)
        model.load_state_dict(obj)
        if device == "cuda":
            model = model.half()                    # fp16 权重
        print(f"[加载] state_dict 格式 -> 重建模型完成")
    else:                                           # 整对象格式(int8 量化)
        model = obj
        if device == "cuda":
            model = model.to(device)                # 注意: int8 量化模型不支持 CUDA, 请用 cpu
        print(f"[加载] 整对象格式(量化模型) -> 直接加载完成")
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description="评估量化/fp16 模型并保存日志")
    parser.add_argument("--model", required=True, help="模型文件路径")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                        help="运行设备(auto: 有 GPU 用 cuda, 否则 cpu)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"模型文件不存在: {model_path}")
        return
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    print(f"模型: {model_path} | 设备: {device}")

    # 1.加载测试集
    _, _, test_dataloader = build_dataloader()
    print(f"测试集已加载: {len(test_dataloader.dataset)} 条")

    # 2.加载模型
    model = load_model(model_path, device)

    # 3.评估
    f1, report, confmat = model2eval(test_dataloader, model, device)
    lines = [
        "=" * 60,
        f"评估时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"模型文件: {model_path}",
        f"运行设备: {device}",
        f"测试集: {len(test_dataloader.dataset)} 条(均衡 1:1:1:1)",
        f"weighted_f1: {f1:.4f}",
        "分类评估报告:",
        report,
        "混淆矩阵:",
        str(confmat),
        "=" * 60,
    ]
    body = "\n".join(lines)
    print(body)

    # 4.保存日志
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{model_path.stem}_eval_{datetime.now():%Y%m%d_%H%M}.txt"
    log_path.write_text(body, encoding="utf-8")
    print(f"\n日志已保存: {log_path}")


if __name__ == "__main__":
    main()
