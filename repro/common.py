"""可复现数据契约与统一绝对路径。"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path) -> tuple[tuple[str, int], ...]:
    rows: list[tuple[str, int]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not row["text"].strip() or type(row.get("label")) is not int or row["label"] not in range(4):
                raise ValueError(f"{path.name}:{number} 需要非空text和0到3的整数label")
            rows.append((row["text"], row["label"]))
    if not rows:
        raise ValueError("数据集不能为空")
    return tuple(rows)
