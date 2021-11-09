'''
Author: jianzhnie
Date: 2021-11-09 14:50:15
LastEditTime: 2021-11-09 15:09:29
LastEditors: jianzhnie
Description: 

'''
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification


class TextModel():
    def __init__(configautoml):
    # Download configuration from huggingface.co and cache.
    config = AutoConfig.from_pretrained('bert-base-cased')
    model = AutoModelForSequenceClassification.from_config(config)