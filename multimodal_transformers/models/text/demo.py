'''
Author: jianzhnie
Date: 2021-11-09 14:40:19
LastEditTime: 2021-11-09 14:41:29
LastEditors: jianzhnie
Description: 

'''
import transformers
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
print(model)
