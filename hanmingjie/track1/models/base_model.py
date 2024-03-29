import torch
from torch import nn
from torch.nn import functional as F

from pytorch_transformers.modeling_bert import BertForPreTraining, BertConfig, BertModel, BertEmbeddings
from pytorch_transformers.modeling_roberta import RobertaModel

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_type = config['encoder_type']
        if self.encoder_type == 'bert':
            self.encoder = BertModel.from_pretrained(config['encoder_path'])
        elif self.encoder_type == 'roberta':
            self.encoder = RobertaModel.from_pretrained(config['encoder_path'])

        self.classifier = nn.Linear(self.encoder.config.hidden_size, 3)
        self.relu = nn.ReLU()

    def forward(self, batch):
        point = batch['point_tensor']
        point_mask = batch['point_mask']
        sentence = batch['sentence_tensor']
        sentence_mask = batch['sentence_mask']
        # label = batch['label']

        cat_feat = torch.cat([point, sentence], dim=-1)
        cat_mask = torch.cat([point_mask, sentence_mask], dim=-1)
        # cat_feat = point
        # cat_mask = point_mask
        mul_out = self.encoder(cat_feat, attention_mask=cat_mask)

        mul_features = mul_out[1]

        mul_features = self.relu(mul_features)
        cls_out = self.classifier(mul_features)
        pred_prob = F.softmax(cls_out, dim=-1)

        loss_input = {
            'pred': pred_prob,
            'label': batch['gt_label']
        }

        pred = cls_out.argmax(dim=-1)

        pred_w_index = torch.cat([batch['index'].unsqueeze(-1), pred.unsqueeze(-1)], dim=-1)

        return loss_input, pred_w_index




