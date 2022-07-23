import torch
from torch import nn
from torch.nn import functional as F

from pytorch_transformers.modeling_bert import BertForPreTraining, BertConfig, BertModel, BertEmbeddings


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert_config = BertConfig()
        self.bert_embedding = BertEmbeddings(self.bert_config)
        self.bert = BertModel.from_pretrained(config['bert_path'])

        self.classifier = nn.Linear(self.bert.config.hidden_size, 3)
        self.relu = nn.ReLU()

    def forward(self, batch):
        point = batch['point_tensor']
        point_mask = batch['point_mask']
        sentence = batch['sentence_tensor']
        sentence_mask = batch['sentence_mask']
        # label = batch['label']

        cat_feat = torch.cat([point, sentence], dim=-1)
        cat_mask = torch.cat([point_mask, sentence_mask], dim=-1)
        mul_out = self.bert(cat_feat, attention_mask=cat_mask)

        mul_features = mul_out[1]

        mul_features = self.relu(mul_features)
        cls_out = self.classifier(mul_features)
        pred_prob = F.softmax(cls_out, dim=-1)

        pred = cls_out.argmax(dim=-1)

        pred_w_index = torch.cat([batch['index'].unsqueeze(-1), pred.unsqueeze(-1)], dim=-1)

        return pred_prob, pred_w_index




