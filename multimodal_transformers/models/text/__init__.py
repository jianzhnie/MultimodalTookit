'''
Author: jianzhnie
Date: 2021-11-08 17:11:49
LastEditTime: 2021-11-17 09:59:05
LastEditors: jianzhnie
Description:

'''

from .deeptext import BertTextModel, BertWithTabular
from .deeptext_auto import AutoModelWithText

__all__ = ['BertWithTabular', 'AutoModelWithText', 'BertTextModel']
