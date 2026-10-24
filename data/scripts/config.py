from pathlib import Path

# 所有路径锚定到 data/ 目录(本文件位于 data/scripts/ 下),与运行目录无关
DATA_DIR = Path(__file__).resolve().parent.parent


class Config:
    def __init__(self):
        self.raw_data_path = DATA_DIR / "danmaku_raw.jsonl"
        self.figs_dir = DATA_DIR / "figs"
        self.pilot_sample_path = DATA_DIR / "pilot_sample.jsonl"
        self.pilot_sample_size = 500   # 试标注样本量
        self.seed = 42
        self.pilot_labeled_path = DATA_DIR / "pilot_labeled.jsonl"   # LLM 标注结果
        self.review_sample_path = DATA_DIR / "review_sample.jsonl"   # 人工抽检样本
