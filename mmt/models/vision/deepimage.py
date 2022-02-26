'''
Author: jianzhnie
Date: 2021-11-10 18:22:22
LastEditTime: 2022-02-25 18:58:03
LastEditors: jianzhnie
Description:

'''
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models


class ImageEncoder(nn.Module):

    def __init__(self, is_require_grad=True):
        super(ImageEncoder, self).__init__()
        self.is_require_grad = is_require_grad
        # Resnet Encoder
        self.resnet_encoder = self.build_encoder()
        # Flatten the feature map grid [B, D, H, W] --> [B, D, H*W]
        self.flatten = nn.Flatten(start_dim=1, end_dim=3)
        self.output_dim = 2048

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        r"""Forward pass connecting the `'backbone'` with the `'head layers'`"""
        x_feat = self.resnet_encoder(x)
        x_feat = self.flatten(x_feat)
        return x_feat

    def build_encoder(self):
        """Builds the backbone layers."""
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        resnet_encoder = nn.Sequential(*modules)
        for param in resnet_encoder.parameters():
            param.requires_grad = self.is_require_grad
        return resnet_encoder


if __name__ == '__main__':
    x = torch.ones(32, 3, 224, 224)
    model = ImageEncoder(is_require_grad=True)
    output = model(x)
    print(output.shape)
