
import torch
from torch import nn
from torch.nn import functional as F

class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.score_k = config['score_k']
        hidden_size = config['hidden_size']
        fc_mid = hidden_size * config['fc_dim_times']
        # linear
        self.fc_1 = nn.Sequential(
            nn.Linear(hidden_size, fc_mid),
            nn.Linear(fc_mid, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, fc_mid),
            nn.Linear(fc_mid, hidden_size),
            nn.Tanh()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(hidden_size, fc_mid),
            nn.Linear(fc_mid, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, fc_mid),
            nn.Linear(fc_mid, hidden_size),
            nn.Tanh()
        )

    def forward(self, batch):
        if self.training:
            point_tensor = batch['point_tensor']
            point_mask = batch['point_mask']
            sentence_pos_tensor = batch['sentence_pos_tensor']
            sentence_pos_mask = batch['sentence_pos_mask']
            sentence_neg_tensor_list = batch['sentence_neg_tensor_list']
            sentence_neg_mask_list = batch['sentence_neg_mask_list']

            point_fc_out = self.fc_1(point_tensor)
            sentence_pos_fc_out = self.fc_2(sentence_pos_tensor)
            sentence_neg_fc_out = [self.fc_2(neg_tensor).unsqueeze(1) for neg_tensor in sentence_neg_tensor_list]
            sentence_neg_fc_out = torch.cat(sentence_neg_fc_out, dim=1)
            loss_input = {
                'query': point_fc_out,
                'positive_key': sentence_pos_fc_out,
                'negative_keys': sentence_neg_fc_out
            }

            pred = torch.zeros(point_tensor.size(0)).to(device=point_tensor.device)
            pred_w_index = torch.cat([pred.unsqueeze(-1), pred.unsqueeze(-1)], dim=-1)

            return loss_input, pred_w_index
        else:
            index = batch['index']
            point_tensor = batch['point_tensor']
            point_mask = batch['point_mask']
            sentence_tensor = batch['sentence_tensor']
            sentence_mask = batch['sentence_mask']

            point_fc_out = self.fc_1(point_tensor)
            sentence_fc_out = self.fc_2(sentence_tensor)

            query, positive_key = normalize(point_fc_out, sentence_fc_out)
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            # point_fc_out = F.normalize(point_fc_out, dim=-1)
            # sentence_fc_out = F.normalize(sentence_fc_out, dim=-1)
            # positive_logit = torch.sum(point_fc_out * sentence_fc_out, dim=1, keepdim=True)
            # score = torch.abs(positive_logit).squeeze(0)
            score = positive_logit.squeeze(0)

            # pred_prob = torch.cat([torch.gt(score, self.score_k).unsqueeze(-1), torch.le(score, self.score_k).unsqueeze(-1)], dim=-1)
            loss_input = None
            pred_w_index = torch.cat([index.unsqueeze(-1), torch.gt(score, self.score_k).unsqueeze(-1)], dim=-1)
            return loss_input, pred_w_index

def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]