'''
Author: jianzhnie
Date: 2021-11-10 18:22:22
LastEditTime: 2021-11-11 15:50:09
LastEditors: jianzhnie
Description:

'''
import torch.nn as nn
from torch import Tensor
from torchvision import models


class ImageCoAttentionEncoder(nn.Module):

    def __init__(self, is_require_grad):
        super(ImageCoAttentionEncoder, self).__init__()

        self.is_require_grad = is_require_grad

        # Resnet Encoder
        self.resnet_encoder = self.build_encoder()

        # Flatten the feature map grid [B, D, H, W] --> [B, D, H*W]
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x_img):
        x_feat_map = self.resnet_encoder(x_img)

        # Flatten (16 x 16 x 2048) --> (16*16, 2048)
        x_feat = self.flatten(x_feat_map)

        x_feat = x_feat.permute(0, 2, 1)  # [batch_size, spatial_locs, 2048]

        return x_feat

    def build_encoder(self):
        """
        Given Resnet backbone, build the encoder network from all layers except the last 2 layers.

        :return: model (nn.Module)
        """
        resnet = models.resnet152(pretrained=True)

        modules = list(resnet.children())[:-2]

        resnet_encoder = nn.Sequential(*modules)

        for param in resnet_encoder.parameters():
            param.requires_grad = self.is_require_grad

        return resnet_encoder


class DeepImage(nn.Module):
    def __init__(self, is_require_grad=True):
        super(DeepImage, self).__init__()
        self.is_require_grad = is_require_grad
        # Resnet Encoder
        self.resnet_encoder = self.build_encoder()
        # Flatten the feature map grid [B, D, H, W] --> [B, D, H*W]
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        r"""Forward pass connecting the `'backbone'` with the `'head layers'`"""
        x_feat = self.resnet_encoder(x)
        # Flatten (16 x 16 x 2048) --> (16*16, 2048)
        x_feat = self.flatten(x_feat)
        x_feat = x_feat.permute(0, 2, 1)  # [batch_size, spatial_locs, 2048]
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
    model = DeepImage(is_require_grad=True)
    print(model)