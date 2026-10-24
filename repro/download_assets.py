"""下载最终发布权重并验证 SHA-256；不需要 GitHub 密钥。"""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from repro.common import ROOT, resolve_path, sha256_file


def download_weight(destination: Path, url: str, expected_sha256: str) -> Path:
    if destination.exists() and sha256_file(destination) == expected_sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "danmaku-classifier-reproduction"})
    with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(block)
    if sha256_file(temporary) != expected_sha256:
        raise ValueError("权重校验失败，请不要使用下载的文件")
    temporary.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "deploy/model/bert_classifier_fp16_model.pt"))
    args = parser.parse_args()
    manifest = json.loads((ROOT / "reports/source_manifest.json").read_text(encoding="utf-8"))
    weight = manifest["weights"]
    url = "https://github.com/Squanch1/danmaku-classifier/releases/download/v1.0.0/" + weight["name"]
    path = download_weight(resolve_path(args.output), url, weight["sha256"])
    print(f"权重下载并校验成功：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
