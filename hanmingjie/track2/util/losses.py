from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.nn import functional as F
from info_nce import InfoNCE
class BaseLoss:
    def __init__(self, config):
        self.config = config
        self.loss_name = config['loss_name']
        if self.loss_name == 'CrossEntropyLoss':
            self.loss = CrossEntropyLoss()
        elif self.loss_name == 'CrossEntropyLossWeighted':
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            weight = torch.Tensor(config['weight']).to(device=device)
            self.loss = CrossEntropyLoss(weight)
        elif self.loss_name == 'MSELoss':
            self.label_tran = lambda x: x.to(torch.float32)
            self.loss = MSELoss()
        elif self.loss_name == "InfoNCE":
            self.loss = InfoNCE(negative_mode=config['negative_mode'])

    def __call__(self, *args, **kwargs):
        assert self.loss is not None
        if self.loss_name == "MSELoss":
            kwargs['label'] = self.label_tran(kwargs['label'])
        if self.loss_name == "InfoNCE":
            query = kwargs['query']
            positive_key = kwargs['positive_key']
            negative_keys = kwargs['negative_keys']
            return self.loss(query, positive_key, negative_keys)
        else:
            pred = kwargs['pred']
            label = kwargs['label']
            return self.loss(pred, label)
        
    