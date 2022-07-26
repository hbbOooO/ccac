import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from track1.processers.base_processer import Pipeline

# load bert features from files directly
class SimpleDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_type = config['dataset_type']
        assert self.dataset_type == 'train' or self.dataset_type == 'val' or self.dataset_type == 'inference'
        self.data_path = config['data_root_dir'] + '/' + config['data_file_path']
        if self.dataset_type == 'train':
            self._read_pair()
        else:
            self._read()
        self.pipeline = Pipeline(config['pipeline'])

    def _read(self):
        with open(self.data_path, encoding='utf-8') as f:
            lines = f.readlines()
        data = []
        points = []
        for i in range(len(lines)):
            line = lines[i]
            point, sentence, label = line.split('\t')
            label = label[:-1]
            if point not in points:
                points.append(point)
            data.append({
                'index': i,
                'point': point,
                'point_index': points.index(point),
                'sentence': sentence,
                'label': label
            })
        self.data = self._clean(data)
        self.gt_label = {item['index']: abs(int(item['label'])) for item in self.data}

    def _clean(self, data):
        # remove emelemt of data when point is equal to sentence
        return [item for item in data if item['point'] != item['sentence']]
    
    def _read_pair(self):
        with open(self.data_path, encoding='utf-8') as f:
            lines = f.readlines()
        data = []
        points = []
        for i in range(len(lines)):
            line = lines[i]
            point, sentence, label = line.split('\t')
            label = label[:-1]
            if point not in points:
                points.append(point)
            data.append({
                'index': i,
                'point': point,
                'point_index': points.index(point),
                'sentence': sentence,
                'label': label
            })
        data = self._clean(data)

        group_data = []
        point_index_list = []
        for item in data:
            index, point, point_index, sentence, label = item['index'], item['point'], item['point_index'], item['sentence'], item['label']
            if point_index not in point_index_list:
                point_index_list.append(point_index)
                sentence_pos = []
                sentence_neg = []
                if label == '0': sentence_neg.append(index)
                else: sentence_pos.append(index)
                group_data.append({
                    'point_index': point_index,
                    'sentence_pos_index': sentence_pos,
                    'sentence_neg_index': sentence_neg
                })
            else:
                # point_index = point_index_llst[point_index]
                if label == '0': group_data[point_index]['sentence_neg_index'].append(index)
                else: group_data[point_index]['sentence_pos_index'].append(index)
        
        # pos/neg = 1/10
        pair_data = []
        for item in group_data:
            point_index = item['point_index']
            sentence_pos_index = item['sentence_pos_index']
            sentence_neg_index = item['sentence_neg_index']
            neg_select = [0 for _ in sentence_neg_index]
            if len(sentence_pos_index) == 0: continue
            if len(sentence_neg_index) < 10: continue
            while not all(neg_select):
                for pos_index in sentence_pos_index:
                    neg_ids = np.random.choice(len(sentence_neg_index), 10, replace=False)
                    neg_index_list = [sentence_neg_index[i] for i in range(len(sentence_neg_index)) if i in neg_ids]
                    neg_select = [neg_select[i] if i not in neg_ids else 1 for i in range(len(neg_select))]
                    pair_data.append({
                        'point_index': point_index,
                        'sentence_pos_index': pos_index,
                        'sentence_neg_index': neg_index_list
                    })
        self.data = pair_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sample = {}
        return self.pipeline(item, sample)

    def evaluate(self, prediction):
        gt_label = self.gt_label
        gt_label = {key: gt_label[key] for key in prediction.keys()}
        # sort
        gt_label_sorted = [gt_label[k] for k in sorted(gt_label.keys())]
        prediction_sorted = [prediction[k] for k in sorted(prediction.keys())]

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