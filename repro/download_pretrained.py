"""下载独立的原始 BERT 预训练权重，仅重新训练时需要。"""

import argparse

from huggingface_hub import snapshot_download

from repro.common import ROOT, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="google-bert/bert-base-chinese")
    parser.add_argument("--output", default=str(ROOT / "bert/bert-base-chinese"))
    args = parser.parse_args()
    output = resolve_path(args.output)
    snapshot_download(repo_id=args.repo_id, local_dir=str(output),
                      allow_patterns=["config.json", "model.safetensors", "vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
    print(f"预训练模型下载到：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
