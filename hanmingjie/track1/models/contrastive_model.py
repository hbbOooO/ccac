import torch
from torch import nn
from torch.nn import functional as F

from pytorch_transformers.modeling_bert import BertConfig, BertModel, BertEmbeddings
from pytorch_transformers.modeling_roberta import RobertaModel

class ContrastiveModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Bert
        if config['encoder_type'] == 'bert':
            self.encoder_1 = BertModel.from_pretrained(config['encoder_path'])
            self.encoder_2 = BertModel.from_pretrained(config['encoder_path'])
        elif config['encoder_type'] == 'roberta':
            self.encoder_1 = RobertaModel.from_pretrained(config['encoder_path'])
            self.encoder_2 = RobertaModel.from_pretrained(config['encoder_path'])
        
        fc_mid = self.encoder_1.config.hidden_size * config['fc_dim_times']
        # linear
        self.fc_1 = nn.Sequential(
            nn.Linear(self.encoder_1.config.hidden_size, fc_mid),
            nn.Linear(fc_mid, self.encoder_1.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.encoder_1.config.hidden_size, fc_mid),
            nn.Linear(fc_mid, self.encoder_1.config.hidden_size),
            nn.Tanh()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(self.encoder_1.config.hidden_size, fc_mid),
            nn.Linear(fc_mid, self.encoder_1.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.encoder_1.config.hidden_size, fc_mid),
            nn.Linear(fc_mid, self.encoder_1.config.hidden_size),
            nn.Tanh()
        )
        # frozen
        self._frozen_enoder(config['frozen_encoder'])


    def _frozen_enoder(self, frozen=True):
        if frozen:
            frozen_names = ['encoder_1', 'encoder_2']
            for name, param in self.named_parameters():
                for bert_name in frozen_names:
                    if bert_name in name:
                        param.requires_grad = False

    def _pool(self, feat, mask):
        mask_expanded = mask.unsqueeze(-1).expand(feat.size()).float()
        sum_embeddings = torch.sum(feat * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        output_vectors = []
        output_vectors.append(sum_embeddings / sum_mask)
        embedding = torch.cat(output_vectors, 1)
        return embedding

    # A series of operations
    def _get_fc_out_1(self, feat, mask):
        # encode
        out = self.encoder_1(feat, attention_mask=mask)[0]
        # pool
        embedding = self._pool(out, mask)
        # linear
        fc_out = self.fc_1(embedding)
        return fc_out

    # A series of operations
    def _get_fc_out_2(self, feat, mask):
        # encode
        out = self.encoder_2(feat, attention_mask=mask)[0]
        # pool
        embedding = self._pool(out, mask)
        # linear
        fc_out = self.fc_2(embedding)
        return fc_out
        

    def forward(self, batch):
        index = batch['index']
        point = batch['point_tensor']
        point_mask = batch['point_mask']
        sentence_pos = batch['sentence_pos_tensor']
        sentence_pos_mask = batch['sentence_pos_mask']
        sentence_neg = batch['sentence_neg_tensor']
        sentence_neg_mask = batch['sentence_neg_mask']

        point_fc_out = self._get_fc_out_1(point, point_mask)
        sentence_pos_fc_out = self._get_fc_out_2(sentence_pos, sentence_pos_mask)
        sentence_neg_fc_out = []
        for neg, neg_mask in zip(sentence_neg, sentence_neg_mask):
            sentence_neg_fc_out.append(self._get_fc_out_2(neg, neg_mask).unsqueeze(1))
        sentence_neg_fc_out = torch.cat(sentence_neg_fc_out, dim=1)

        loss_input = {
            'query': point_fc_out,
            'positive_key': sentence_pos_fc_out,
            'negative_keys': sentence_neg_fc_out
        }

        pred = torch.zeros(point.size(0)).to(device=point.device)
        pred_w_index = torch.cat([index.unsqueeze(-1), pred.unsqueeze(-1)], dim=-1)

        return loss_input, pred_w_index

    





