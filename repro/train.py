"""固定公开数据与配置的重新训练入口；不覆盖原实验日志。"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from repro.common import ROOT, load_dataset, resolve_path
from repro.model import BertClassifier


def encode_batch(batch: list[tuple[str, int]], tokenizer: BertTokenizer, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = tokenizer([text for text, _ in batch], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device), torch.tensor([label for _, label in batch], device=device)


def validation_score(model: BertClassifier, loader: DataLoader, tokenizer: BertTokenizer, device: str) -> float:
    actual: list[int] = []
    predicted: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            ids, mask, labels = encode_batch(batch, tokenizer, device)
            predicted.extend(model(ids, mask).argmax(dim=-1).cpu().tolist())
            actual.extend(labels.cpu().tolist())
    return float(f1_score(actual, predicted, average="weighted"))


def preserve_rows(batch: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return batch


def train(pretrained: Path, output: Path, epochs: int, seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = BertTokenizer.from_pretrained(str(pretrained), local_files_only=True)
    model = BertClassifier(pretrained, pretrained=True).to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(load_dataset(ROOT / "bert/processed_data/train.jsonl"), batch_size=32, shuffle=True, generator=generator, collate_fn=preserve_rows)
    val_loader = DataLoader(load_dataset(ROOT / "bert/processed_data/val.jsonl"), batch_size=32, collate_fn=preserve_rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    loss_function = torch.nn.CrossEntropyLoss()
    output.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    logs: list[dict] = []
    step = 0
    for epoch in range(epochs):
        model.train()
        for index, batch in enumerate(train_loader):
            ids, mask, labels = encode_batch(batch, tokenizer, device)
            optimizer.zero_grad()
            loss = loss_function(model(ids, mask), labels)
            loss.backward()
            optimizer.step()
            step += 1
            if (index + 1) % 25 == 0 or index + 1 == len(train_loader):
                score = validation_score(model, val_loader, tokenizer, device)
                if score > best_score:
                    best_score = score
                    torch.save(model.state_dict(), output / "best_model.pt")
                record = dict(epoch=epoch + 1, step=step, loss=float(loss.detach()), validation_weighted_f1=score)
                logs.append(record)
                print(json.dumps(record), flush=True)
                model.train()
    torch.save(model.state_dict(), output / "last_model.pt")
    (output / "training_run.json").write_text(json.dumps(dict(seed=seed, epochs=epochs, learning_rate=3e-5, batch_size=32, max_length=128, device=device, best_validation_weighted_f1=best_score, records=logs), indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", default=str(ROOT / "bert/bert-base-chinese"))
    parser.add_argument("--output", default=str(ROOT / "repro_output"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs 必须大于0")
    pretrained = resolve_path(args.pretrained)
    if not (pretrained / "model.safetensors").exists() and not (pretrained / "pytorch_model.bin").exists():
        parser.error("缺少原始 BERT 预训练权重，请先运行 python -m repro.download_pretrained")
    train(pretrained, resolve_path(args.output), args.epochs, args.seed, "cuda" if torch.cuda.is_available() else "cpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
