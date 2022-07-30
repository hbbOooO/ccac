# change .ckpt file to .pth file
import sys
import torch 
sys.path.append('/root/autodl-nas/ccac/hanmingjie')

# ckpt = torch.load('/root/autodl-tmp/save/ccac/track1/baseline/model/epoch_7.ckpt')
ckpt = torch.load('/root/autodl-nas/ccac/hanmingjie/submit/track2/epoch_14.ckpt')
model_weight = ckpt['model']
torch.save(model_weight, '/root/autodl-nas/ccac/hanmingjie/submit/track2/model_weight.pth')

