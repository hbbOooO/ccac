from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from torch.nn import functional as F
import torch
import sys 
import gzip



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
        if len(processor_config) == 0:
            processors.append(lambda x: x)
        self.processors = processors
    
    def __call__(self, item, sample):
        processors = self.processors
        for processor in processors:
            processor(item, sample)
        return sample

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
        self.point_max_length = config['point_max_length']
        self.sentence_max_length = config['sentence_max_length']
        self.bert_tokenizer_config = config
        self.bert_tokenizer = BertTokenizer.from_pretrained(config['bert_tokenizer_path'])
        self.pad_id = self.bert_tokenizer.pad_token_id

    def _tokenize(self, input, max_length):
        indices = self.bert_tokenizer.encode(input, add_special_tokens=True)
        tensor = torch.Tensor([self.pad_id for _ in range(max_length)]).to(dtype=torch.long)
        length = len(indices) if len(indices) < max_length else max_length
        tensor[:length] = torch.Tensor(indices)[:length]
        mask = torch.Tensor([1 if i < length else 0 for i in range(max_length)])
        return tensor, mask

    def __call__(self, item, sample):
        point = item['point']
        sentence = item['sentence']

        point_tensor, point_mask = self._tokenize(point, self.point_max_length)
        sentence_tensor, sentence_mask = self._tokenize(sentence, self.sentence_max_length)

        sample['point_tensor'] = point_tensor
        sample['point_mask'] = point_mask
        sample['sentence_tensor'] = sentence_tensor
        sample['sentence_mask'] = sentence_mask

        sample['index'] = item['index']
    
class RobertaProcessor:
    def __init__(self, config):
        self.config = config
        self.point_max_length = config['point_max_length']
        self.sentence_max_length = config['sentence_max_length']
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(config['roberta_tokenizer_path'])
        self.pad_id = self.roberta_tokenizer.pad_token_id

    def _tokenize(self, input, max_length):
        indices = self.roberta_tokenizer.encode(input, add_special_tokens=True)
        tensor = torch.Tensor([self.pad_id for _ in range(max_length)]).to(dtype=torch.long)
        length = len(indices) if len(indices) < max_length else max_length
        tensor[:length] = torch.Tensor(indices)[:length]
        mask = torch.Tensor([1 if i < length else 0 for i in range(max_length)])
        return tensor, mask

    def __call__(self, item, sample):
        point = item['point']
        sentence = item['sentence']

        point_tensor, point_mask = self._tokenize(point, self.point_max_length)
        sentence_tensor, sentence_mask = self._tokenize(sentence, self.sentence_max_length)

        sample['point_tensor'] = point_tensor
        sample['point_mask'] = point_mask
        sample['sentence_tensor'] = sentence_tensor
        sample['sentence_mask'] = sentence_mask

        sample['index'] = item['index']

class LabelProcessor:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, item, sample):
        label = int(item['label'])
        sample['gt_label'] = label + 1

        return item

class ContrastiveBertProcessor:
    def __init__(self, config):
        self.config = config
        self.point_max_length = config['point_max_length']
        self.sentence_pos_max_length = config['sentence_pos_max_length']
        self.sentence_neg_max_length = config['sentence_neg_max_length']
        self.bert_tokenizer_config = config
        self.bert_tokenizer = BertTokenizer.from_pretrained(config['bert_tokenizer_path'])
        self.pad_id = self.bert_tokenizer.pad_token_id

    def _tokenize(self, input, max_length):
        indices = self.bert_tokenizer.encode(input, add_special_tokens=True)
        tensor = torch.Tensor([self.pad_id for _ in range(max_length)]).to(dtype=torch.long)
        length = len(indices) if len(indices) < max_length else max_length
        tensor[:length] = torch.Tensor(indices)[:length]
        mask = torch.Tensor([1 if i < length else 0 for i in range(max_length)])
        return tensor, mask

    def __call__(self, item, sample):
        point = item['point']
        sentence_pos = item['sentence_pos']
        sentence_neg = item['sentence_neg']

        point_tensor, point_mask = self._tokenize(point, self.point_max_length)
        sentence_pos_tensor, sentence_pos_mask = self._tokenize(sentence_pos, self.sentence_pos_max_length)
        neg_res = [self._tokenize(sentence_neg[i], self.sentence_neg_max_length) for i in range(len(sentence_neg))]
        sentence_neg_tensor = [item[0] for item in neg_res]
        sentence_neg_mask = [item[1] for item in neg_res]

        sample['index'] = item['index']

        sample['point_tensor'] = point_tensor
        sample['point_mask'] = point_mask
        sample['sentence_pos_tensor'] = sentence_pos_tensor
        sample['sentence_pos_mask'] = sentence_pos_mask
        sample['sentence_neg_tensor'] = sentence_neg_tensor
        sample['sentence_neg_mask'] = sentence_neg_mask


class ContrastiveRobertaProcessor:
    def __init__(self, config):
        self.config = config
        self.point_max_length = config['point_max_length']
        self.sentence_pos_max_length = config['sentence_pos_max_length']
        self.sentence_neg_max_length = config['sentence_neg_max_length']
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(config['roberta_tokenizer_path'])
        self.pad_id = self.roberta_tokenizer.pad_token_id

    def _tokenize(self, input, max_length):
        indices = self.roberta_tokenizer.encode(input, add_special_tokens=True)
        tensor = torch.Tensor([self.pad_id for _ in range(max_length)]).to(dtype=torch.long)
        length = len(indices) if len(indices) < max_length else max_length
        tensor[:length] = torch.Tensor(indices)[:length]
        mask = torch.Tensor([1 if i < length else 0 for i in range(max_length)])
        return tensor, mask

    def __call__(self, item, sample):
        point = item['point']
        sentence_pos = item['sentence_pos']
        sentence_neg = item['sentence_neg']

        point_tensor, point_mask = self._tokenize(point, self.point_max_length)
        sentence_pos_tensor, sentence_pos_mask = self._tokenize(sentence_pos, self.sentence_pos_max_length)
        neg_res = [self._tokenize(sentence_neg[i], self.sentence_neg_max_length) for i in range(len(sentence_neg))]
        sentence_neg_tensor = [item[0] for item in neg_res]
        sentence_neg_mask = [item[1] for item in neg_res]

        sample['index'] = item['index']

        sample['point_tensor'] = point_tensor
        sample['point_mask'] = point_mask
        sample['sentence_pos_tensor'] = sentence_pos_tensor
        sample['sentence_pos_mask'] = sentence_pos_mask
        sample['sentence_neg_tensor'] = sentence_neg_tensor
        sample['sentence_neg_mask'] = sentence_neg_mask


class DualLabelProcessor:

    def __init__(self, config):
        self.config = config
    
    def __call__(self, item, sample):
        sample['gt_label'] = abs(int(item['label']))

class SimpleProcessor:
    def __init__(self, config):
        self.config = config
        self.run_type = config['run_type']
        self.feat_root_dir = config['feat_root_dir']
        # self.encoder_type = config['encoder_type']
        
    def __call__(self, item, sample):
        point_dir = self.feat_root_dir + 'point/'
        sentence_dir = self.feat_root_dir + 'sentence/'
        if self.run_type == 'train':
            point_index = item['point_index']
            sentence_pos_index = item['sentence_pos_index']
            sentence_neg_index = item['sentence_neg_index']
            point_pth = torch.load(gzip.GzipFile(point_dir+str(point_index)+'.pth.gz', "rb"))
            point_tensor = point_pth['point_tensor']
            point_mask = point_pth['point_mask']
            sentence_pos_pth = torch.load(gzip.GzipFile(sentence_dir+str(sentence_pos_index)+'.pth.gz', "rb"))
            sentence_pos_tensor = sentence_pos_pth['sentence_tensor']
            sentence_pos_mask = sentence_pos_pth['sentence_mask']
            sentence_neg_tensor_list = []
            sentence_neg_mask_list = []
            for neg_index in sentence_neg_index:
                neg_pth = torch.load(gzip.GzipFile(sentence_dir+str(neg_index)+'.pth.gz', "rb"))
                sentence_neg_tensor_list.append(neg_pth['sentence_tensor'])
                sentence_neg_mask_list.append(neg_pth['sentence_mask'])
            sample['point_tensor'] = point_tensor
            sample['point_mask'] = point_mask
            sample['sentence_pos_tensor'] = sentence_pos_tensor
            sample['sentence_pos_mask'] = sentence_pos_mask
            sample['sentence_neg_tensor_list'] = sentence_neg_tensor_list
            sample['sentence_neg_mask_list'] = sentence_neg_mask_list
        else:
            point_index = item['point_index']
            sentence_index = item['index']
            point_pth = torch.load(gzip.GzipFile(point_dir+str(point_index)+'.pth.gz', "rb"))
            point_tensor = point_pth['point_tensor']
            point_mask = point_pth['point_mask']
            sentence_pth = torch.load(gzip.GzipFile(sentence_dir+str(sentence_index)+'.pth.gz', "rb"))
            sentence_tensor = sentence_pth['sentence_tensor']
            sentence_mask = sentence_pth['sentence_mask']
            sample['index'] = sentence_index
            sample['point_tensor'] = point_tensor
            sample['point_mask'] = point_mask
            sample['sentence_tensor'] = sentence_tensor
            sample['sentence_mask'] = sentence_mask


class StageTwoProcessor:
    def __init__(self, config):
        self.config = config

    def __call__(self, item, sample):
        sample['gt_label'] = int(item['label'])


