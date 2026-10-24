import pandas as pd


def processing_data(path):
    raw_data = pd.read_json(path, lines=True)
    df = pd.DataFrame(raw_data[['text', 'label']])
    df["label"] = df['label'].apply(lambda x: x - 1)
    df.to_json('../processed_data/val.jsonl', orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    processing_data("../raw_data/val_data.jsonl")

