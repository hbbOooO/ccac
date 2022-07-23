from asyncio.log import logger
import torch
import os

from common.logger import Logger

class CheckPoint:
    def __init__(self, config):
        self.config = config
        self.ckpt_dir = config['ckpt_dir']
        self.save_by_epoch = config['save_by_epoch']
        self.save_best = config['save_best']
        self.save_last = config['save_last']
        self.save_interval = config['save_interval']
    
    def save(self, epoch, model, optimizer, ccac_config, val_metric=None, best_epoch=None, best_val_metric=None, val=True):
        if val:
            ckpt = {
                'epoch': epoch,
                'val_metric': val_metric,
                'best_epoch': best_epoch,
                'best_val_metric': best_val_metric, 
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ccac_config': ccac_config
            }
        else:
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ccac_config': ccac_config
            }
        ckpt_dir = self.ckpt_dir
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        if self.save_by_epoch and epoch % self.save_interval == 0:
            ckpt_path = ckpt_dir + 'epoch_' + str(epoch) + '.ckpt'
            Logger.get_logger().info('save checkpoint in {}'.format(ckpt_path))
            torch.save(ckpt, ckpt_path)
        if self.save_best and epoch == best_epoch:
            ckpt_path = ckpt_dir + 'epoch_best.ckpt'
            Logger.get_logger().info('save checkpoint in {}'.format(ckpt_path))
            torch.save(ckpt, ckpt_path)
        if  self.save_last:
            ckpt_path = ckpt_dir + 'epoch_last.ckpt'
            Logger.get_logger().info('save checkpoint in {}'.format(ckpt_path))
            torch.save(ckpt, ckpt_path)
    
    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        return ckpt