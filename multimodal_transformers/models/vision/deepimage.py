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
from .model_zoo import get_model, get_model_list, get_model_input_size
import torch.nn as nn
from timm.models.layers.classifier import ClassifierHead


class DeepImage(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    

    

