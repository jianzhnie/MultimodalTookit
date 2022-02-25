'''
Author: jianzhnie
Date: 2021-12-06 11:57:53
LastEditTime: 2021-12-06 12:00:32
LastEditors: jianzhnie
Description:

'''
import io

import requests
import torch
import torch.nn as nn


def load_model(path: str, device: torch.device = None) -> nn.Module:
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path)
        resp.raise_for_status()

        with io.BytesIO(resp.content) as buf:
            return torch.load(buf, map_location=device)
    else:
        with open(path, 'rb') as f:
            return torch.load(f, map_location=device)
