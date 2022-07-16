import torch
import os

class CheckPoint:
    def __init__(self, config):
        self.config = config
        self.ckpt_dir = config['ckpt_dir']
        self.save_by_epoch = config['save_by_epoch']
        self.save_best = config['save_best']
    
    def save(self, epoch, val_metric, best_epoch, best_val_metric, model, optimizer, ccac_config, ):
        ckpt = {
            'epoch': epoch,
            'val_metric': val_metric,
            'best_epoch': best_epoch,
            'best_val_metric': best_val_metric, 
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ccac_config': ccac_config
        }
        ckpt_dir = self.ckpt_dir
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        if self.save_by_epoch:
            ckpt_path = ckpt_dir + 'epoch_' + str(epoch) + '.ckpt'
            torch.save(ckpt, ckpt_path)
        if self.save_best and epoch == best_epoch:
            ckpt_path = ckpt_dir + 'epoch_best.ckpt'
            torch.save(ckpt, ckpt_path)
    
    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        return ckpt