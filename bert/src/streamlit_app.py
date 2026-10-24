"""
弹幕质量分级 · Streamlit 前端页面

功能:
- 选择不同模型(BERT fp32 / fp16 / qint8 / 蒸馏 BiLSTM)进行推理
- 手动输入弹幕(每行一条)批量预测
- 一键「随机 1 万条弹幕推理」: 从原始弹幕集随机采样 10000 条并推理
- 显示: 总弹幕数量 / 总推理耗时 / 单条推理耗时
基于现有代码: 复用 config.py / bert_classifier.py, 等级说明读取 raw_data/class.txt。

启动: streamlit run streamlit_app.py
"""
import json
import random
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from config import Config
from bert_classifier import TMFBertClassifer

st.set_page_config(page_title="弹幕质量分级", page_icon=":material/comment:", layout="wide")

# 项目根目录(锚定, 任何位置运行都找得到各模型权重)
ROOT = Path(__file__).resolve().parent.parent.parent
# 原始弹幕集
DANMAKU_RAW = ROOT / "data" / "danmaku_raw.jsonl"

conf = Config()
# config.py 里 device 是字符串("cuda"/"cpu"), 这里统一转成 torch.device
DEVICE = torch.device(conf.device)

# ---------------- 各模型加载 ----------------

def _load_bert_state_dict(path: Path, half: bool = False):
    """按 state_dict 加载 BERT 分类模型(fp32 或 fp16)。"""
    model = TMFBertClassifer().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    if half and DEVICE.type == "cuda":
        model = model.half()
    return model.eval()


def build_bilstm_from_sd(sd: dict) -> torch.nn.Module:
    """按 state_dict 实际尺寸构建 BiLSTM(自动适配训练时的隐藏维度, 不依赖 config)。

    蒸馏模型可能用不同维度训练过(128/256...), 硬编码尺寸会导致 load_state_dict
    报 size mismatch。这里从权重形状反推结构, 任何版本都能加载。
    """
    embed_dim = sd["embed.weight"].shape[1]                                   # 词嵌入维度
    # 只匹配 weight_ih_l<数字>(双向 LSTM 还有 *_reverse 键, 需排除)
    import re
    num_layers = max(int(m.group(1)) for k in sd if (m := re.search(r"weight_ih_l(\d+)$", k))) + 1
    hidden = sd["lstm.weight_ih_l0"].shape[0] // 4                            # 单向隐藏维度
    fc_in = sd["fc.weight"].shape[1]                                          # fc 输入(双向时 = hidden*2)
    dropout = 0.2 if num_layers > 1 else 0.0

    class _BILSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(conf.vocab_size, embed_dim)
            self.lstm = torch.nn.LSTM(embed_dim, hidden, num_layers,
                                      bidirectional=True, dropout=dropout)
            self.fc = torch.nn.Linear(fc_in, conf.class_num)

        def forward(self, input_ids, attention_mask):
            embed = self.embed(input_ids)
            attention_mask = attention_mask.unsqueeze(dim=-1)
            input_ = (embed * attention_mask).transpose(0, 1)
            lstm_out, _ = self.lstm(input_)
            lstm_out = lstm_out.transpose(0, 1)
            mask = attention_mask
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).squeeze(-1)
            pooled = (lstm_out * mask).sum(dim=1) / lengths
            return self.fc(pooled)

    return _BILSTM()


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    """按名字加载模型(每个模型进程内只加载一次)。返回 (model, device)。"""
    with st.spinner(f"正在加载模型 {name}…"):
        if name == "BERT-base (fp32)":
            return _load_bert_state_dict(ROOT / "bert/save_model/bert_classifier_best_model.pt"), DEVICE
        if name == "BERT-base (fp16)":
            return _load_bert_state_dict(
                ROOT / "bert_quantization/save_model/bert_classifier_fp16_model.pt", half=True), DEVICE
        if name == "BERT-base (qint8)":
            # qint8 动态量化模型: 整模型保存, 只能在 CPU 上运行
            model = torch.load(ROOT / "bert_quantization/save_model/bert_classifier_qint8_model.pt",
                               map_location="cpu", weights_only=False)
            return model.eval(), torch.device("cpu")
        if name == "蒸馏 BiLSTM 学生":
            path = ROOT / "bert_distll/save_model/bilstm_distill_best_model.pt"
            sd = torch.load(path, map_location=DEVICE)
            model = build_bilstm_from_sd(sd).to(DEVICE)   # 按权重尺寸自适应构建
            model.load_state_dict(sd)
            return model.eval(), DEVICE
        raise ValueError(f"未知模型: {name}")


def predict_batch(texts: list[str], model, device: torch.device,
                  batch_size: int = 256, progress=None) -> tuple[list[dict], float]:
    """批量预测: 返回 (每条结果列表, 总耗时ms)。

    加速关键: 按长度排序后分批推理, 再按原顺序还原结果。
    padding="longest" 会把整批 pad 到批内最长文本 —— 弹幕平均 ~13 字,
    若随机批里混入一条 100 字长弹幕, 整批都被迫算到 100 字(浪费 ~7 倍计算)。
    长度相近的文本同批后, padding 浪费大幅降低, CPU(qint8)推理可快 3~5 倍。
    """
    total = len(texts)
    order = sorted(range(total), key=lambda i: len(texts[i]))   # 按长度升序的索引
    results = [None] * total
    t0 = time.perf_counter()
    for start in range(0, total, batch_size):
        idxs = order[start:start + batch_size]
        chunk = [texts[i] for i in idxs]
        enc = conf.tokenizer(chunk, return_tensors="pt", max_length=conf.max_len,
                             padding="longest", truncation=True)
        with torch.no_grad():
            logits = model(enc.input_ids.to(device), enc.attention_mask.to(device))
            probs = torch.softmax(logits, dim=-1)
            pred_idxs = torch.argmax(probs, dim=-1)
        for k, i in enumerate(idxs):
            idx = int(pred_idxs[k].item())
            results[i] = {"pred_class": conf.class_list[idx], "prob": float(probs[k, idx].item())}
        if progress is not None:
            done = min(start + len(idxs), total)
            progress.progress(done / total, text=f"推理中… {done}/{total}")
    total_ms = (time.perf_counter() - t0) * 1000
    return results, total_ms


# ---------------- 原始弹幕集 ----------------

@st.cache_data(show_spinner=False)
def load_raw_texts() -> list[str]:
    """读取原始弹幕集全部文本(缓存, 只读一次)。"""
    with open(DANMAKU_RAW, "r", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def sample_and_predict(model, device: torch.device, n: int = 10000, batch_size: int = 256,
                       progress=None) -> tuple[list[dict], list[str], float]:
    """接口: 从原始弹幕集随机取 n 条并推理。返回 (每条结果列表, 对应文本, 总耗时ms)。"""
    all_texts = load_raw_texts()
    sample = random.sample(all_texts, min(n, len(all_texts)))
    results, total_ms = predict_batch(sample, model, device, batch_size, progress)
    return results, sample, total_ms


# 等级说明: 从 class.txt 读取, 与训练标注规范保持同步
LEVEL_NAMES, LEVEL_DESCS = {}, {}
with open(conf.class_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.split()
        name = parts[0]  # "1级"
        LEVEL_NAMES[name] = parts[1] if len(parts) > 1 else ""
        LEVEL_DESCS[name] = line.split("定义:", 1)[1].strip() if "定义:" in line else ""

# ---------------- 页面 ----------------
MODELS = ["BERT-base (fp32)", "BERT-base (fp16)", "BERT-base (qint8)", "蒸馏 BiLSTM 学生"]

st.title("弹幕质量分级")
st.caption("选择模型 → 输入弹幕(每行一条)批量预测, 或一键随机 1 万条推理 · 1级 垃圾/违规 · 2级 低质 · 3级 普通 · 4级 优质")

# 模型选择(放在表单外, 手动预测与随机采样共用)
model_name = st.selectbox("推理模型", MODELS)

# ---- 手动批量预测 ----
with st.form("predict_form", border=False):
    text = st.text_area("弹幕内容(每行一条)", height=200, key="danmaku_text",
                        placeholder="例如:\n前方高能预警\n哈哈哈哈哈哈\n66666")
    submitted = st.form_submit_button("批量预测", type="primary", icon=":material/rate_review:")

if submitted:
    texts = [t.strip() for t in text.splitlines() if t.strip()]
    if not texts:
        st.warning("请输入至少一条弹幕")
    else:
        with st.spinner(f"正在用 {model_name} 推理 {len(texts)} 条弹幕…"):
            model, device = load_model(model_name)
            results, total_ms = predict_batch(texts, model, device)
        avg_ms = total_ms / len(texts)

        col1, col2, col3 = st.columns(3)
        col1.metric("总弹幕数量", f"{len(texts)} 条")
        col2.metric("总推理耗时", f"{total_ms / 1000:.2f} s")
        col3.metric("单条推理耗时", f"{avg_ms:.1f} ms")

        df = pd.DataFrame({
            "弹幕": texts,
            "等级": [r["pred_class"] for r in results],
            "名称": [LEVEL_NAMES.get(r["pred_class"], "") for r in results],
            "置信度": [f"{r['prob']:.2%}" for r in results],
        })
        st.dataframe(df, width="stretch")

# ---- 随机 1 万条推理 ----
st.divider()
st.subheader("随机采样推理")
st.caption(f"从原始弹幕集({DANMAKU_RAW})随机获取 10000 条并推理, 进度见进度条")

if st.button("随机 1 万条弹幕推理", type="primary", icon=":material/shuffle:"):
    with st.spinner(f"正在用 {model_name} 推理 1 万条弹幕…"):
        model, device = load_model(model_name)
        progress = st.progress(0, text="开始推理…")
        results, sample, total_ms = sample_and_predict(model, device, n=10000, progress=progress)
        progress.empty()
    avg_ms = total_ms / len(sample)

    col1, col2, col3 = st.columns(3)
    col1.metric("总弹幕数量", f"{len(sample)} 条")
    col2.metric("总推理耗时", f"{total_ms / 1000:.2f} s")
    col3.metric("单条推理耗时", f"{avg_ms:.1f} ms")

    dist = pd.Series([r["pred_class"] for r in results]).value_counts().reindex(conf.class_list, fill_value=0)
    dist_df = pd.DataFrame({
        "等级": dist.index,
        "名称": [LEVEL_NAMES.get(l, "") for l in dist.index],
        "条数": dist.values,
        "占比": [f"{v / len(results):.2%}" for v in dist.values],
    })
    st.dataframe(dist_df, width="stretch")
