from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
from track2.models.contrastive_model import ContrastiveModel

class ContrastiveClassificationHead(ContrastiveModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.score_k = config['score_k']

    def forward(self, batch):
        index = batch['index']
        point = batch['point_tensor']
        point_mask = batch['point_mask']
        sentence = batch['sentence_tensor']
        sentence_mask = batch['sentence_mask']

        point_fc_out = self._get_fc_out_1(point, point_mask).unsqueeze(1)
        sentence_fc_out = self._get_fc_out_2(sentence, sentence_mask).unsqueeze(-1)

        score = torch.bmm(point_fc_out, sentence_fc_out).view(point.size(0))
        score = torch.abs(score / sqrt(self.encoder_1.config.hidden_size))
        # score = torch.cosine_similarity(point_fc_out, sentence_fc_out).view(point.size(0))
        pred_prob = torch.cat([torch.le(score, self.score_k).unsqueeze(-1), torch.gt(score, self.score_k).unsqueeze(-1)], dim=-1)
        loss_input = {
            'pred': pred_prob,
            'label': batch['gt_label']
        }
        pred_w_index = torch.cat([index.unsqueeze(-1), torch.ge(score, self.score_k).unsqueeze(-1)], dim=-1)

        return loss_input, pred_w_index
        



