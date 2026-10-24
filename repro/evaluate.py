"""验证 Release 权重，不下载基础预训练权重、不调用外部模型服务。"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import transformers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import BertTokenizer

from repro.common import ROOT, load_dataset, resolve_path, sha256_file
from repro.model import BertClassifier


def evaluate(checkpoint: Path, dataset: Path, device: str, batch_size: int = 32) -> dict:
    rows = load_dataset(dataset)
    config_path = ROOT / "deploy" / "model" / "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(str(config_path), local_files_only=True)
    model = BertClassifier(config_path)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    model.eval()
    predictions: list[int] = []
    start = perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            encoded = tokenizer([text for text, _ in rows[offset:offset + batch_size]], padding="max_length",
                                truncation=True, max_length=128, return_tensors="pt")
            logits = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    labels = [label for _, label in rows]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "samples": len(rows),
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "precision": "fp16" if device == "cuda" else "fp32", "max_length": 128, "batch_size": batch_size,
        "checkpoint_sha256": sha256_file(checkpoint), "dataset_sha256": sha256_file(dataset),
        "accuracy": accuracy_score(labels, predictions), "weighted_f1": f1_score(labels, predictions, average="weighted"),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "classification_report": classification_report(labels, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2, 3]).tolist(),
        "evaluation_seconds": round(perf_counter() - start, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "deploy/model/bert_classifier_fp16_model.pt"))
    parser.add_argument("--dataset", default=str(ROOT / "bert/processed_data/test.jsonl"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default=str(ROOT / "reports/reproduced_fp16.json"))
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error("缺少发布权重，请先运行 python -m repro.download_assets")
    report = evaluate(checkpoint, resolve_path(args.dataset), device)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("samples", "device", "accuracy", "weighted_f1", "macro_f1", "evaluation_seconds")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
