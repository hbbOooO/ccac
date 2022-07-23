from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

class Optimizer:
    def __init__(self, config, parameters):
        self.config = config
        self.use_warmup = config['use_warmup']
        if self.use_warmup:
            self.wamup_iter = config['wamup_iter']
        else:
            self.wamup_iter = None
        self.lr = config['lr']
        self.lr_step = config['lr_step']
        self.optimizer_name = config['optimizer_name']

        if self.optimizer_name == 'sgd' or self.optimizer_name == 'SGD':
            self.optimizer = SGD(
                parameters,
                lr=self.lr
            )
        self.scheduler = LambdaLR(self.optimizer, lambda step: scheduler_fun(step, self.use_warmup, self.wamup_iter, self.lr_step))

def scheduler_fun(step, use_warmup, warmup_iter, lr_step):
    if isinstance(step, int):
        epoch, iter = 0, 0
    else:
        epoch, iter = step
    if use_warmup:
        if iter < warmup_iter:
            return iter / warmup_iter
    from bisect import bisect
    idx = bisect(lr_step, epoch)
    return pow(0.1, idx)
