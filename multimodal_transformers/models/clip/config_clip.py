'''
Author: jianzhnie
Date: 2021-12-03 11:42:11
LastEditTime: 2021-12-03 17:14:22
LastEditors: jianzhnie
Description:

'''
import torch


class ClipConfig():
    model_type = 'clip'
    debug = False
    image_path = '/media/robin/DATA/datatsets/image_data/clickr8k/Images'
    captions_path = '/media/robin/DATA/datatsets/image_data/clickr8k'
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_model = 'resnet18'
    image_embedding = 512
    text_encoder_model = 'distilbert-base-uncased'
    text_embedding = 768
    text_tokenizer = 'distilbert-base-uncased'
    max_length = 200

    pretrained = True  # for both image encoder and text encoder
    trainable = True  # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1
