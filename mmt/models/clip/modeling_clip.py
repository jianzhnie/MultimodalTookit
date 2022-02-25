'''
Author: jianzhnie
Date: 2021-12-03 11:48:26
LastEditTime: 2021-12-03 17:54:27
LastEditors: jianzhnie
Description:

'''

import timm
import torch
import torch.nn.functional as F
from torch import nn
from transformers import DistilBertConfig, DistilBertModel

from .config_clip import ClipConfig as cfg


class ImageEncoder(nn.Module):
    """Encode images to a fixed size vector."""

    def __init__(self, model_name, pretrained, trainable):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool='avg')
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):

    def __init__(self, model_name, pretrained, trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):

    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):

    def __init__(
        self,
        image_model=cfg.image_model,
        text_model=cfg.text_encoder_model,
        temperature=cfg.temperature,
        image_embedding=cfg.image_embedding,
        text_embedding=cfg.text_embedding,
        projection_dim=cfg.projection_dim,
        dropout=cfg.dropout,
        pretrained=cfg.pretrained,
        trainable=cfg.trainable,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            model_name=image_model, pretrained=pretrained, trainable=trainable)
        self.text_encoder = TextEncoder(
            model_name=text_model, pretrained=pretrained, trainable=trainable)
        self.image_projection = ProjectionHead(
            embedding_dim=image_embedding,
            projection_dim=projection_dim,
            dropout=dropout)
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding,
            projection_dim=projection_dim,
            dropout=dropout)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch['image'])
        text_features = self.text_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'])
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature,
            dim=-1)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()


if __name__ == '__main__':
    image_encoder = ImageEncoder(
        model_name=cfg.image_model,
        pretrained=cfg.pretrained,
        trainable=cfg.trainable)
    print(image_encoder)
    clp_model = CLIPModel()
    print(clp_model)

    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print(loss)
