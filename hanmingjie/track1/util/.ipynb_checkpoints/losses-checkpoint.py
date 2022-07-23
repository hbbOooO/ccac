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
        elif self.loss_name == "MSELossWeighted":
            self.label_tran = lambda x: x.to(torch.float32)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            weight = torch.Tensor(config['weight']).to(device=device)
            self.loss = MSELoss()
        self.loss = InfoNCE(weight)
        self.pred_tran = getattr(self, 'pred_tran', lambda x: x)
        self.label_tran = getattr(self, 'label_tran', lambda x: x)

    def __call__(self, pred, label):
        assert self.loss is not None
        pred = self.pred_tran(pred)
        label = self.label_tran(label)
        return self.loss(pred, label)
        
    