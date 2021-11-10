'''
Author: jianzhnie
Date: 2021-11-09 14:40:19
LastEditTime: 2021-11-10 18:18:00
LastEditors: jianzhnie
Description: 

'''
import torch.nn as nn

from transformers import AutoModelForSequenceClassification


class AutoTextModel(nn.Module):

    def __init__(self, args) -> None:
        self.args = args

        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.text_model, num_labels=args.num_classes)

    def forward(self, X):
        out = self.model(X)
        return out
