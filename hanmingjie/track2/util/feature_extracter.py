import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from pytorch_transformers.modeling_bert import BertModel
from pytorch_transformers.modeling_roberta import RobertaModel
from tqdm import tqdm
import os
import numpy as np
import gzip

# Hyper parametr
# data_root_dir = '/root/autodl-tmp/data/Track1/'
# data_file_names = ['train.txt', 'dev.txt', 'test.txt']
output_dir = '/root/autodl-tmp/data/Track2/feat/'
model_type = 'bert'
output_dir += model_type + '/'
run_types = ['mixed']
dataset_config = [
    {
        'data_root_dir': '/root/autodl-tmp/data/Track2/',
        'data_file_name': 'mixed.txt',
        'point_max_length': 512,
        'sentence_max_length': 512,
        'tokenizer_path': 'bert-base-uncased'
    }
    # ,{
    #     'data_root_dir': '/root/autodl-tmp/data/Track2/',
    #     'data_file_name': 'dev.txt',
    #     'point_max_length': 512,
    #     'sentence_max_length': 512,
    #     'tokenizer_path': 'bert-base-uncased'
    # },{
    #     'data_root_dir': '/root/autodl-tmp/data/Track2/',
    #     'data_file_name': 'test.txt',
    #     'point_max_length': 512,
    #     'sentence_max_length': 512,
    #     'tokenizer_path': 'bert-base-uncased'
    # }
]

model_config = {
    'encoder_path': 'bert-base-uncased',
    'frozen_encoder': True,
}

class ExtractDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_path = config['data_root_dir'] + config['data_file_name']
        self.point_max_length = config['point_max_length']
        self.sentence_max_length = config['sentence_max_length']
        if model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config['tokenizer_path'])
        elif model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(config['tokenizer_path'])
        self.pad_id = self.tokenizer.pad_token_id
        self._read()

    def _read(self):
        with open(self.data_path, encoding='utf-8') as f:
            lines = f.readlines()
        data = []
        point_indices = []
        points = []
        for i in range(len(lines)):
            line = lines[i]
            point, sentence, label = line.split('\t')
            if point not in points:
                points.append(point)
                point_indices.append(len(point_indices))
            label = label[:-1]
            data.append({
                'index': i,
                'point': point,
                'point_index': point_indices[-1],
                'sentence': sentence,
                'label': label
            })
        self.data = data

    def _tokenize(self, input, max_length):
        indices = self.tokenizer.encode(input, add_special_tokens=True)
        tensor = torch.Tensor([self.pad_id for _ in range(max_length)]).to(dtype=torch.long)
        length = len(indices) if len(indices) < max_length else max_length
        tensor[:length] = torch.Tensor(indices)[:length]
        mask = torch.Tensor([1 if i < length else 0 for i in range(max_length)])
        return tensor, mask
    
    def _preprocess(self, item):
        index = item['index']
        point = item['point']
        point_index = item['point_index']
        sentence = item['sentence']
        point_tensor, point_mask = self._tokenize(point, self.point_max_length)
        sentence_tensor, sentence_mask = self._tokenize(sentence, self.sentence_max_length)

        sample = {
            'index': index,
            'point_tensor': point_tensor,
            'point_mask': point_mask,
            'point_index': point_index,
            'sentence_tensor': sentence_tensor,
            'sentence_mask': sentence_mask
        }
        return sample

    def __getitem__(self, index):
        return self._preprocess(self.data[index])

    def __len__(self):
        return len(self.data)



class ExtractModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Bert
        if model_type == 'bert':
            self.encoder_1 = BertModel.from_pretrained(config['encoder_path'])
            self.encoder_2 = BertModel.from_pretrained(config['encoder_path'])
        elif model_type == 'roberta':
            self.encoder_1 = RobertaModel.from_pretrained(config['encoder_path'])
            self.encoder_2 = RobertaModel.from_pretrained(config['encoder_path'])
        
        # frozen
        self._frozen_enoder(config['frozen_encoder'])

    def _frozen_enoder(self, frozen=True):
        if frozen:
            frozen_names = ['encoder_1', 'encoder_2']
            for name, param in self.named_parameters():
                for bert_name in frozen_names:
                    if bert_name in name:
                        param.requires_grad = False

    # batch size = 1
    def forward(self, batch, run_type):
        index = batch['index']
        point = batch['point_tensor']
        point_mask = batch['point_mask']
        point_index = batch['point_index']
        sentence = batch['sentence_tensor']
        sentence_mask = batch['sentence_mask']

        point_out = self.encoder_1(point, attention_mask=point_mask)[1]
        sentence_out = self.encoder_1(sentence, attention_mask=sentence_mask)[1]

        # save bert featue
        sentence_save_out = {
            'index': index.squeeze(0),
            'sentence_tensor': sentence_out.squeeze(0),
            'sentence_mask': sentence_mask.squeeze(0)
        }
        sentence_out_dir = output_dir+run_type+'/sentence/'
        if not os.path.exists(sentence_out_dir): os.makedirs(sentence_out_dir)
        torch.save(sentence_save_out, gzip.GzipFile(sentence_out_dir+str(index.item())+'.pth.gz', 'wb'))

        point_out_dir = output_dir + run_type + '/point/'
        if not os.path.exists(point_out_dir+str(point_index.item())+'.pth.gz'):
            point_save_out = {
                'index': index.squeeze(0),
                'point_index': point_index.squeeze(0),
                'point_tensor': point_out.squeeze(0),
                'point_mask': point_mask.squeeze(0)
            }
            if not os.path.exists(point_out_dir): os.makedirs(point_out_dir)
            torch.save(point_save_out, gzip.GzipFile(point_out_dir + str(point_index.item()) + '.pth.gz', 'wb'))
        # np.savez(output_dir+run_type+'/'+str(index.item())+'.pth', save_tensor)

def _to_cuda(batch):
    if isinstance(batch, list):
        for i in range(len(batch)): batch[i] = _to_cuda(batch[i])
    if isinstance(batch, dict):
        fields = batch.keys()
        for field in fields:
            batch[field] = _to_cuda(batch[field])
    if isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    return batch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataloaders = []
for config in dataset_config:
    dataset = ExtractDataset(config)
    dataloaders.append(DataLoader(dataset,batch_size=1))

model = ExtractModel(model_config)
model.train()
model.to(device=device)


for run_type, dataloader in zip(run_types, dataloaders):
    for batch in tqdm(dataloader):
        _to_cuda(batch)
        model(batch, run_type)





