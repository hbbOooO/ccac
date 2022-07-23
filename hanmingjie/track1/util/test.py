

# with open('/root/autodl-tmp/data/Track1/train.txt') as f:
#     lines = f.readlines()

# print(sum([1 for line in lines if line.split('\t')[-1][:-1] == '0']))
# print(sum([1 for line in lines if line.split('\t')[-1][:-1] == '1']))
# print(sum([1 for line in lines if line.split('\t')[-1][:-1] == '-1']))

# import torch
# from torch.nn import functional as F

# def softmax(X):
#     """
#     :param X:
#     :return:
#     """
#     # 对向量中的每一个求他 e^x的值
#     X_exp = torch.exp(X)
#     # 对其进行一阶求和
#     partition = X_exp.sum(1, keepdim=True) # 对求出的值任然保持维度
#     return X_exp / partition #这里利用了一个广播机制， 将所有的数都进行除法的运算

# weight = torch.Tensor([5.64015,1.93524e-1,4.76644])
# print(F.softmax(weight, dim=-1))

from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

with open('/root/autodl-tmp/data/Track1/train.txt') as f:
    lines = f.readlines()
samples = []
for line in lines:
    point, sentence, label = line.split('\t')
    label = label[:-1]
    if label == '1' or label == '-1':
        samples.append({
            'point': point,
            'sentence': sentence, 
            'label': int(label)
        })
classifier = pipeline("sentiment-analysis")
pred = []
gt = []
for sample in samples:
    res = classifier(sample['point']+sample['sentence'])
    gt.append(sample['label'])
    p = 1 if res[0]['label'] == 'POSITIVE' else -1
    pred.append(p)
accuracy = accuracy_score(gt, pred)
precision = precision_score(gt, pred, average='macro', zero_division=0)
precision_cls = precision_score(gt, pred, average=None, zero_division=0)
recall = recall_score(gt, pred, average='macro', zero_division=0)
recall_cls = recall_score(gt, pred, average=None, zero_division=0)
f1 = f1_score(gt, pred, average='macro', zero_division=0)
f1_cls = f1_score(gt, pred, average=None, zero_division=0)
print('f1:{:.4f}(avg), {:.4f}(0), {:.4f}(1). precision:{:.4f}(avg), {:.4f}(0), {:.4f}(1). recall:{:.4f}(avg), {:.4f}(0), {:.4f}(1). accuracy:{:.4f}'.format(
    f1,
    f1_cls[0],
    f1_cls[1],
    precision,
    precision_cls[0],
    precision_cls[1],
    recall,
    recall_cls[0],
    recall_cls[1],
    accuracy
))