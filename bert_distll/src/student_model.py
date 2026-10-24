import torch.nn as nn

from config import Config

conf = Config()
class BILSTMClassifier(nn.Module):
    def __init__(self):
        # 继承父类方法
        super().__init__()
        # 词嵌入层
        self.embed = nn.Embedding(conf.vocab_size,conf.embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(conf.embedding_dim, # 词嵌入维度
                            conf.lstm_hidden_size, # 隐藏层的维度
                            conf.num_layer, # 隐藏层的层数
                            bidirectional=True,
                            dropout=conf.dropout)
        # 全连接层
        self.fc = nn.Linear(conf.lstm_hidden_size*2,conf.class_num)

    def forward(self,input_ids,attention_mask):
        # input_ids : [batch_size,seq_len] -->  [batch_size,seq_len,embedding_dim]
        embed = self.embed(input_ids)
        #  attention_mask: [batch_size,seq_len] --> [batch_size,seq_len,1]
        attention_mask= attention_mask.unsqueeze(dim=-1)
        # input : [batch_size,seq_len,embedding_dim] \
        # 目的：将padding位置的词嵌入置为0
        input = embed*attention_mask
        # input : [seq_len,batch_size,embedding_dim]
        input = input.transpose(0,1)
        # lstm_out : [seq_len,batch_size,lstm_hidden_size*2]
        lstm_out,(hn,cn) = self.lstm(input)
        # 有效时间步均值池化(而非取最后时间步):
        # 最后时间步几乎都是 padding(弹幕平均10字, max_len=128), 信息被稀释;
        # 用 attention_mask 只平均真实 token 的输出, 对 padding 鲁棒, 收敛更快
        lstm_out = lstm_out.transpose(0,1)          # [batch_size, seq_len, hidden*2]
        mask = attention_mask                        # [batch_size, seq_len, 1]
        # 注意: lengths 必须是 [B, 1], 不能是 [B, 1, 1]——否则除法会广播出多余维度
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).squeeze(-1)   # [batch_size, 1]
        pooled = (lstm_out * mask).sum(dim=1) / lengths        # [batch_size, hidden*2]
        return self.fc(pooled)


if __name__ == '__main__':
    from utils import build_dataloader
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    model = BILSTMClassifier()
    for input_ids,attention_mask,labels in train_dataloader:

            print(model(input_ids, attention_mask))
            break
