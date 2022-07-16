from torch.nn import CrossEntropyLoss
import torch

class BaseLoss:
    def __init__(self, config):
        self.config = config
        self.loss_name = config['loss_name']
        if self.loss_name == 'CrossEntropyLoss':
            self.loss = CrossEntropyLoss()

    def __call__(self, pred, label):
        assert self.loss is not None
        return self.loss(pred, label)
        
    