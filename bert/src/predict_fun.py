from config import  Config
from bert_classifier import TMFBertClassifer
import torch

# 1.加载配置文件
conf = Config()
# 2.加载模型
model = TMFBertClassifer().to(conf.device)
model.load_state_dict(torch.load(conf.best_model))
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
        input_ids = input.input_ids.to(conf.device)
        attention_mask = input.attention_mask.to(conf.device)
        # 前向传播
        logits = model(input_ids, attention_mask)
        y_pred = torch.argmax(logits,dim=-1)

        data["pred_class"] = conf.class_list[y_pred]
    return data


if __name__ == '__main__':
    print(predict({'text':"哈哈哈你这个也太好笑了"}))