import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from track1.processers.base_processer import Pipeline

class StageTwoDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_type = config['dataset_type']
        assert self.dataset_type == 'train' or self.dataset_type == 'val' or self.dataset_type == 'inference'
        self.data_path = config['data_root_dir'] + '/' + config['data_file_path']
        self._read()
        self.pipeline = Pipeline(config['pipeline'])


    ## read text file
    def _read(self):
        with open(self.data_path, encoding='utf-8') as f:
            lines = f.readlines()
        data = []
        for i in range(len(lines)):
            line = lines[i]
            point, sentence, label = line.split('\t')
            label = label[:-1]
            if label == '0':continue
            if label == '-1': label = '0'
            data.append({
                'index': i,
                'point': point,
                'sentence': sentence,
                'label': label
            })
        # balance
        if self.dataset_type == 'train':
            cls_num = {}
            for item in data:
                label = item['label']
                if label in cls_num.keys():
                    cls_num[label] += 1
                else:
                    cls_num[label] = 1
            # min_num = cls_num['0'] if cls_num['0'] < cls_num['1'] else cls_num['1']
            min_label = '0' if cls_num['0'] < cls_num['1'] else '1'
            expand_num = abs(cls_num['0']-cls_num['1'])
            min_candidates = [item for item in data if item['label']==min_label]
            expand_ids = np.random.choice(len(min_candidates), expand_num, replace=False)
            expand_data = [min_candidates[i] for i in range(len(min_candidates)) if i in expand_ids]
            data.extend(expand_data)


        self.data = data
        self.gt_label = {item['index']: int(item['label']) for item in data}

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