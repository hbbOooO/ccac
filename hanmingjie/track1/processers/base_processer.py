from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.nn import functional as F
import torch
import sys 



class Pipeline:
    def __init__(self, config):
        self.config = config
        self.processors_config = config
        self._init_pipeline()

    def _init_pipeline(self):
        mod = sys.modules[__name__]
        processors_config = self.processors_config
        processors = []
        for processor in processors_config:
            processor_name = processor['name']
            processor_config = processor['config']
            processor_class = getattr(mod, processor_name)
            processor = processor_class(processor_config)
            processors.append(processor)
        self.processors = processors
    
    def __call__(self, item):
        processors = self.processors
        for processor in processors:
            processor(item)
        return item

class BertProcessor:
    def __init__(self, config):
        '''
            the number of words in point and sentence:
            file        point   sentence
            train.txt   13      252
            dev.txt     13      180
            test.txt    13      154
        '''
        self.config = config
        self.point_max_length = 15
        self.sentence_max_length = 260
        self.bert_tokenizer_config = config
        self.bert_tokenizer = BertTokenizer.from_pretrained(config['bert_tokenizer_path'])
        self.pad_id = self.bert_tokenizer.pad_token_id

    def __call__(self, item):
        point = item['point']
        sentence = item['sentence']

        point_indices = self.bert_tokenizer.encode(point)
        sentence_indices = self.bert_tokenizer.encode(sentence)

        point_tensor = torch.Tensor([self.pad_id for _ in range(self.point_max_length)]).to(dtype=torch.long)
        point_len = len(point_indices)
        point_tensor[:point_len] = torch.Tensor(point_indices)
        # point_len = torch.Tensor([point_len])
        point_mask = torch.Tensor([1 if i < point_len else 0 for i in range(self.point_max_length)])

        sentence_tensor = torch.Tensor([self.pad_id for _ in range(self.sentence_max_length)]).to(dtype=torch.long)
        sentence_indices = sentence_indices[:min(len(sentence_indices), self.sentence_max_length)]
        sentence_len = len(sentence_indices)
        sentence_tensor[:sentence_len] = torch.Tensor(sentence_indices)
        # sentence_len = torch.Tensor([sentence_len])
        sentence_mask = torch.Tensor([1 if i < sentence_len else 0 for i in range(self.sentence_max_length)])

        item['point_tensor'] = point_tensor
        item['point_len'] = point_len
        item['point_mask'] = point_mask
        item['sentence_tensor'] = sentence_tensor
        item['sentence_len'] = sentence_len
        item['sentence_mask'] = sentence_mask

        return item
    

class LabelProcessor:
    LABEL_ACCOUNT = 3

    def __init__(self, config):
        self.config = config
    
    def __call__(self, item):
        label = int(item['label'])
        onehot = torch.zeros(self.LABEL_ACCOUNT)
        onehot[label+1] = 1
        item['onehot'] = onehot
        item['gt_label'] = label + 1

        return item



