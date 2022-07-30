# Stance Classification with Target-Specific Neural Attention Networks

import torch
from torch import nn
from torch.nn import functional as F

from pytorch_transformers.modeling_bert import BertForPreTraining, BertConfig, BertModel, BertEmbeddings
from pytorch_transformers.modeling_roberta import RobertaModel

from track1.util.bilstm import BiLSTM

class TAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_type = config['encoder_type']
        # encoder
        if self.encoder_type == 'bert':
            self.mul_encoder = BertModel.from_pretrained(config['encoder_path'])
            self.sentence_encoder = BertModel.from_pretrained(config['encoder_path'])
        elif self.encoder_type == 'roberta':
            self.mul_encoder = RobertaModel.from_pretrained(config['encoder_path'])
            self.sentence_encoder = RobertaModel.from_pretrained(config['encoder_path'])
        # bi-lstm
        # self.bilstm = BiLSTM(**config['lstm'])

        self.att_linear = nn.Linear(self.mul_encoder.config.hidden_size, 1)
        fc_mid = config['fc_mid']
        self.classifier = nn.Sequential(
            nn.Linear(self.mul_encoder.config.hidden_size, fc_mid),
            nn.Linear(fc_mid, self.mul_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.mul_encoder.config.hidden_size, fc_mid),
            nn.Linear(fc_mid, 3)
        )
        

    def forward(self, batch):
        point = batch['point_tensor']
        point_mask = batch['point_mask']
        point_len = batch['point_len']
        sentence = batch['sentence_tensor']
        sentence_mask = batch['sentence_mask']
        sentence_len = batch['sentence_len']

        mul_input = torch.cat([point, sentence], dim=-1)
        mul_mask = torch.cat([point_mask, sentence_mask], dim=-1)
        # cat_feat = point
        # cat_mask = point_mask
        mul_out = self.mul_encoder(mul_input, attention_mask=mul_mask)
        mul_pool_feat = mul_out[1]

        sentence_out = self.sentence_encoder(sentence, attention_mask=sentence_mask)
        sentence_feat = sentence_out[0]
        # lstm_out = self.bilstm(sentence_feat, sentence_len)

        attention_feat = mul_pool_feat.unsqueeze(1).expand(sentence_feat.size())
        score = F.softmax(self.att_linear(attention_feat).squeeze(-1), dim=-1)

        # score = score[:,:lstm_out.size(1)]
        atted_feat = torch.sum(sentence_feat * score.unsqueeze(-1), dim=1)

        pred_prob = F.softmax(self.classifier(atted_feat), dim=-1)

        loss_input = {
            'pred': pred_prob,
            'label': batch['gt_label']
        }

        pred = pred_prob.argmax(dim=-1)

        pred_w_index = torch.cat([batch['index'].unsqueeze(-1), pred.unsqueeze(-1)], dim=-1)

        return loss_input, pred_w_index