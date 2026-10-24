"""
该脚本的目的是: 为BERT预训练模型准备数据, 包含: 原始数据加载, 自定义数据集, 批量处理, 构架数据加载器等功能...
todo 0:加载配置文件和加载分词器
todo 1:定义函数, 实现: 加载并处理原始数据.
函数功能: 从指定文本文件中加载数据, 处理成'文本内容-标签索引'的元组列表, 供后续数据集封装使用.
    :param file_path: 原始数据文件路径. 文件内容(每行数据)格式为: '文本\t标签'
    :return:
        list: 元素为元组形式, 格式为: [('文本标题', '标签索引'), ('文本标题', '标签索引'), ...]
        例如: [('体验2D 巅峰 倚天屠龙记十大创新概览', 8), ('六月触发考研“多米诺”--统考专业课指导', 3), ...]

todo 2. 自定义数据集类.
自定义的数据集类, 封装"文本-标签"数据, 使PyTorch的DataLoader能批量加载.
我们自定义的数据集类的作用: 将原始数据(load_raw_data函数的输出结果) 转换为DataLoader可识别的格式.

todo 3. 定义批数据处理函数 collate_fn
函数作用: 对DataLoader的一个批次(batch)原始数据进行预处理, 例如: 分词, 填充, 转张量.
    :param batch: 格式为: [(标题1, 分类索引), (标题2, 分类索引), (text3, label3)...]
    :return: 包含3个PyTorch张量的元组
        input_ids: 分词后token的id, 形状为: (batch_size, max_length)
        attention_mask: 注意力掩码(标记有效token 和 填充token), 形状和 input_ids一致.
        labels: 批次标签, 形状是: (batch_size, )

todo 4. 构建数据加载器
函数作用: 构建训练集, 测试集, 验证集的数据加载器, 供模型训练, 评估使用.
    :return: 包含3个DataLoader的元组, 顺序为: (train_dataloader, dev_dataloader, test_dataloader)


"""
import json

import torch

from config import Config
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# todo 0:加载配置文件
conf = Config()


# todo 1:定义函数, 实现: 加载并处理原始数据.
def load_raw_data(file_path):
    """
    函数功能: 从指定 JSONL 文件中加载数据, 处理成'文本内容-标签索引'的元组列表, 供后续数据集封装使用.
    :param file_path: 原始数据文件路径. 文件内容(每行)格式为: {"text": "...", "label": 0~3}
    注意: processed_data 的 jsonl 标签已是 0~3(PyTorch 交叉熵要求标签从 0 开始), 无需再转换.
    :return:
        list: 元素为元组形式, 格式为: [('文本', 标签索引0~3), ('文本', 标签索引0~3), ...]
    """
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Loading data'):
            # 1.去除行首行尾的空格
            line = line.strip()
            # 2.去除空行
            if not line:
                continue
            # 3.获取文本和标签值(jsonl 格式, 标签已为 0~3)
            line = json.loads(line)
            text, label = line['text'], line['label']
            data.append((text, int(label)))
    return data


# todo 2. 自定义数据集类.
class TMFDataset(Dataset):
    """
自定义的数据集类, 封装"文本-标签"数据, 使PyTorch的DataLoader能批量加载.
我们自定义的数据集类的作用: 将原始数据(load_raw_data函数的输出结果) 转换为DataLoader可识别的格式.
    """

    def __init__(self, data):
        """
        :param data: 原始数据, 格式为: [('文本标题', '标签索引'), ('文本标题', '标签索引'), ...]
        """

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


# todo 3. 定义批数据处理函数 collate_fn
def collate_fn(batch):
    """
    函数作用: 对DataLoader的一个批次(batch)原始数据进行预处理, 例如: 分词, 填充, 转张量.
        :param batch: 格式为: [(标题1, 分类索引), (标题2, 分类索引), (text3, label3)...]
        :return: 包含3个PyTorch张量的元组
            input_ids: 分词后token的id, 形状为: (batch_size, max_length)
            attention_mask: 注意力掩码(标记有效token 和 填充token), 形状和 input_ids一致.
            labels: 批次标签, 形状是: (batch_size, )
    """
    # 1.获取文本列表和标签列表
    texts = [items[0] for items in batch]
    labels = [items[1] for items in batch]
    # 2.文本转向量
    input = conf.tokenizer(texts,
                           padding="max_length",
                           max_length=conf.max_len,
                           truncation=True,
                           return_tensors="pt")
    input_ids = input["input_ids"]
    attention_mask = input["attention_mask"]
    labels = torch.tensor(labels)
    return input_ids, attention_mask, labels


# todo 4. 构建数据加载器
def build_dataloader():
    """
    函数作用: 构建训练集, 测试集, 验证集的数据加载器, 供模型训练, 评估使用.
    :return: 包含3个DataLoader的元组, 顺序为: (train_dataloader, dev_dataloader, test_dataloader)
    """
    # 1.获取原始数据
    train_data = load_raw_data(conf.train_data_path)
    dev_data = load_raw_data(conf.val_data_path)
    test_data = load_raw_data(conf.test_data_path)

    # 2.构建数据集对象
    train_dataset = TMFDataset(train_data)
    dev_dataset = TMFDataset(dev_data)
    test_dataset = TMFDataset(test_data)
    # 3.构建数据加载器
    # collate_fn 是将每个batch送给自定义函数进行预处理 然后将处理后数据返回
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == '__main__':
    print(load_raw_data(conf.train_data_path)[:5])

    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    for idx,(input_ids, attention_mask, labels) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
        print(input_ids)
        print(attention_mask)
        print(labels)
        break
    print(conf.tokenizer(["我是你爸"]))
