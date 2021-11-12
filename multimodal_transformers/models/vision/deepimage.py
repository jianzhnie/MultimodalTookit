'''
Author: jianzhnie
Date: 2021-11-10 18:22:22
LastEditTime: 2021-11-12 14:48:29
LastEditors: jianzhnie
Description:

'''
import torch
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
        """Given Resnet backbone, build the encoder network from all layers
        except the last 2 layers.

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


POOLING_BREAKDOWN = {
    1: (1, 1),
    2: (2, 1),
    3: (3, 1),
    4: (2, 2),
    5: (5, 1),
    6: (3, 2),
    7: (7, 1),
    8: (4, 2),
    9: (3, 3),
}


class ImageEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        model = models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(
            POOLING_BREAKDOWN[args.num_image_embeds])

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048


if __name__ == '__main__':
    model = DeepImage(is_require_grad=True)
    print(model)
