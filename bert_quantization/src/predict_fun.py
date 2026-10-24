from config import  Config
from bert_classifier import TMFBertClassifer
import torch

# 1.加载配置文件
conf = Config()
# 2.加载模型
# 注意 因为模型保存的时候没有只保存参数字典，顾这里加载模型不能只加载参数字典，weights_only=False
model = torch.load(conf.quantization_model_path,weights_only=False)
model.eval()
# 3.模型的预测
def predict(data:dict):
    # 训推一致
    input = conf.tokenizer(data["text"],
                           return_tensors='pt',
                           max_length=conf.max_len,
                           padding="max_length"
                           )
    with torch.no_grad():
        # input_ids = input.input_ids.to(conf.device)
        # attention_mask = input.attention_mask.to(conf.device)
        # 前向传播
        logits = model(input.input_ids, input.attention_mask)
        y_pred = torch.argmax(logits,dim=-1)

        data["pred_class"] = conf.class_list[y_pred]
    return data


if __name__ == '__main__':
    print(predict({'text':"名师辅导：2012考研英语虚拟语气三种用法"}))