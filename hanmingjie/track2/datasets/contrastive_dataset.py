import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import numpy as np

from track2.processers.base_processer import Pipeline

class ContrastiveDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_type = config['dataset_type']
        assert self.dataset_type == 'train' or self.dataset_type == 'val' or self.dataset_type == 'inference'
        self.data_path = config['data_root_dir'] + '/' + config['data_file_path']
        if self.dataset_type == 'train':
            self._read_pair()
        elif self.dataset_type == 'val' or self.dataset_type == 'inference':
            self._read()
        self.pipeline = Pipeline(config['pipeline'])

    # read text file when validating and inference
    def _read(self):
        with open(self.data_path, encoding='utf-8') as f:
            lines = f.readlines()
        data = []
        for i in range(len(lines)):
            line = lines[i]
            point, sentence, label = line.split('\t')
            label = label[:-1]
            data.append({
                'index': i,
                'point': point,
                'sentence': sentence,
                'label': label
            })
        gt_label = {item['index']: abs(int(item['label'])) for item in data}
        self.gt_label = gt_label
        self.data = data


    # read text file when training
    def _read_pair(self):
        with open(self.data_path, encoding='utf-8') as f:
            lines = f.readlines()
        data = []
        sentence_index_llst = {}
        for line in lines:
            point, sentence, label = line.split('\t')
            label = label[:-1]
            if sentence_index_llst.get(point, None) is None:
                index = sentence_index_llst[point] = len(data)
                sentence_pos = []
                sentence_neg = []
                if label == '0': sentence_neg.append(sentence)
                else: sentence_pos.append(sentence)
                data.append({
                    'index': index,
                    'point': point,
                    'sentence_pos': sentence_pos,
                    'sentence_neg': sentence_neg
                })
            else:
                index = sentence_index_llst[point]
                if label == '0': data[index]['sentence_neg'].append(sentence)
                else: data[index]['sentence_pos'].append(sentence)
        
        # pos/neg = 1/10
        pair_data = []
        for item in data:
            sentence_pos = item['sentence_pos']
            sentence_neg = item['sentence_neg']
            neg_select = [0 for _ in sentence_neg]
            if len(sentence_pos) == 0: continue
            if len(sentence_neg) < 10: continue
            while not all(neg_select):
                for pos_item in sentence_pos:
                    neg_ids = np.random.choice(len(sentence_neg), 10, replace=False)
                    neg_list = [sentence_neg[i] for i in range(len(sentence_neg)) if i in neg_ids]
                    neg_select = [neg_select[i] if i not in neg_ids else 1 for i in range(len(neg_select))]
                    pair_data.append({
                        'index': len(pair_data),
                        'point': item['point'],
                        'sentence_pos': pos_item,
                        'sentence_neg': neg_list
                    })
        self.data = pair_data


    def __getitem__(self, index):
        item = self.data[index]
        sample = {}
        return self.pipeline(item, sample)


    def __len__(self):
        return len(self.data)

    def evaluate(self, prediction):
        gt_label = self.gt_label
        gt_label = {key: gt_label[key] for key in prediction.keys()}
        # sort
        gt_label_sorted = [gt_label[k] for k in sorted(gt_label.keys())]
        prediction_sorted = [prediction[k] for k in sorted(prediction.keys())]

        # assert all(item == 0 or item == 1 or item == -1 for item in gt_label_sorted)
        # assert all(item == 0 or item == 1 or item == -1 for item in prediction_sorted)

        accuracy = accuracy_score(gt_label_sorted, prediction_sorted)
        precision = precision_score(gt_label_sorted, prediction_sorted, average='macro', zero_division=0)
        precision_cls = precision_score(gt_label_sorted, prediction_sorted, average=None, zero_division=0)
        recall = recall_score(gt_label_sorted, prediction_sorted, average='macro', zero_division=0)
        recall_cls = recall_score(gt_label_sorted, prediction_sorted, average=None, zero_division=0)
        f1 = f1_score(gt_label_sorted, prediction_sorted, average='macro', zero_division=0)
        f1_cls = f1_score(gt_label_sorted, prediction_sorted, average=None, zero_division=0)

        
        metric = {
            'f1': f1,
            'f1_cls': f1_cls,
            'precision': precision,
            'precision_cls': precision_cls,
            'recall': recall,
            'recall_cls': recall_cls,
            'accuracy': accuracy
        }

        return metric