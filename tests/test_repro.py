from pathlib import Path

import pytest

from repro.common import ROOT, load_dataset, resolve_path, sha256_file


def test_dataset_schema_and_split_isolation() -> None:
    rows = {name: load_dataset(ROOT / "bert" / "processed_data" / f"{name}.jsonl") for name in ("train", "val", "test")}
    assert {name: len(data) for name, data in rows.items()} == {"train": 4000, "val": 1000, "test": 1600}
    texts = {name: {text for text, _ in data} for name, data in rows.items()}
    assert not texts["train"] & texts["val"]
    assert not texts["train"] & texts["test"]
    assert not texts["val"] & texts["test"]
    assert all(sum(label == category for _, label in rows["test"]) == 400 for category in range(4))


def test_path_resolution_is_independent_of_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parent)
    assert resolve_path("bert/processed_data/test.jsonl") == ROOT / "bert" / "processed_data" / "test.jsonl"


def test_published_data_checksums() -> None:
    import json
    manifest = json.loads((ROOT / "reports" / "release_manifest.json").read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["files"] if entry["path"].startswith("bert/processed_data/")]
    assert len(entries) == 3
    assert all(sha256_file(ROOT / entry["path"]) == entry["sha256"] for entry in entries)
