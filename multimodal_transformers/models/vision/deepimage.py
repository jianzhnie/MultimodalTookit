'''
Author: jianzhnie
Date: 2021-11-10 18:22:22
LastEditTime: 2021-11-10 18:51:05
LastEditors: jianzhnie
Description: 

'''

import torch
import torch.nn as nn
import torchvision
import copy
from model_zoo import get_model, get_model_list, get_model_input_size
import torch.nn as nn
from timm.models.layers.classifier import ClassifierHead
from torch import Tensor
from ..tabular.tab_mlp import MLP


class DeepImage(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model_name = args.model_name
        self.freeze_n = args.freeze_n
        self.head_hidden_dims = args.head_hidden_dims
        self.head_activation = args.head_activation
        self.head_dropout = args.head_dropout
        self.head_batchnorm = args.head_batchnorm
        self.head_batchnorm_last = args.head_batchnorm_last
        self.head_linear_first = args.head_linear_first

        vision_model = self.get_model(self.model_name)
        backbone_layers = list(vision_model.children())[:-1]
        self.backbone = self._build_backbone(backbone_layers, self.freeze_n)

        if self.head_hidden_dims is not None:
            assert self.head_hidden_dims[0] == self.output_dim, (
                "The output dimension from the backbone ({}) is not consistent with "
                "the expected input dimension ({}) of the fc-head".format(
                    self.output_dim, self.head_hidden_dims[0]))
            self.imagehead = MLP(
                head_hidden_dims,
                head_activation,
                head_dropout,
                head_batchnorm,
                head_batchnorm_last,
                head_linear_first,
            )
            self.output_dim = self.head_hidden_dims[-1]

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        r"""Forward pass connecting the `'backbone'` with the `'head layers'`"""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        if self.head_hidden_dims is not None:
            out = self.imagehead(x)
            return out
        else:
            return x

    def _build_backbone(self, backbone_layers, freeze_n):
        """
        Builds the backbone layers
        """
        if freeze_n > 8:
            raise ValueError(
                "freeze_n' must be less than or equal to 8 for resnet architectures"
            )
        frozen_layers = []
        trainable_layers = backbone_layers[freeze_n:]
        for layer in backbone_layers[:freeze_n]:
            for param in layer.parameters():
                param.requires_grad = False
            frozen_layers.append(layer)
        trainable_and_frozen_layers = frozen_layers + trainable_layers
        return nn.Sequential(*trainable_and_frozen_layers)