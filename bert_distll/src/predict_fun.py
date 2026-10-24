from config import  Config
from bert_classifier import TMFBertClassifer
from student_model import BILSTMClassifier
import torch

# 1.加载配置文件
conf = Config()
# 2.加载模型
model = BILSTMClassifier()
model.load_state_dict(torch.load(conf.student_best_model))
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
        input_ids = input.input_ids
        attention_mask = input.attention_mask
        # 前向传播
        logits = model(input_ids, attention_mask)
        y_pred = torch.argmax(logits,dim=-1)

        data["pred_class"] = conf.class_list[y_pred]
    return data


if __name__ == '__main__':
    print(predict({'text':"名师辅导：2012考研英语虚拟语气三种用法"}))