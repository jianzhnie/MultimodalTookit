'''
Author: jianzhnie
Date: 2021-12-03 11:31:07
LastEditTime: 2021-12-03 17:29:29
LastEditors: jianzhnie
Description:

'''

import itertools
import sys

import pandas as pd
import torch
from multimodal_transformers.models.clip.config_clip import ClipConfig as cfg
from multimodal_transformers.models.clip.datasets_clip import build_loaders, make_train_valid_dfs
from multimodal_transformers.models.clip.modeling_clip import CLIPModel
from multimodal_transformers.utils.metrics import AverageMeter
from multimodal_transformers.utils.model import get_lr
from tqdm.autonotebook import tqdm
from transformers import DistilBertTokenizer

sys.path.append('../../')


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AverageMeter(name='Train')
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {
            k: v.to(cfg.device)
            for k, v in batch.items() if k != 'caption'
        }
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 'batch':
            lr_scheduler.step()

        count = batch['image'].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(
            train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AverageMeter(name='Valid')

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {
            k: v.to(cfg.device)
            for k, v in batch.items() if k != 'caption'
        }
        loss = model(batch)

        count = batch['image'].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(cfg.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode='train')
    valid_loader = build_loaders(valid_df, tokenizer, mode='valid')

    model = CLIPModel().to(cfg.device)
    params = [{
        'params': model.image_encoder.parameters(),
        'lr': cfg.image_encoder_lr
    }, {
        'params': model.text_encoder.parameters(),
        'lr': cfg.text_encoder_lr
    }, {
        'params':
        itertools.chain(model.image_projection.parameters(),
                        model.text_projection.parameters()),
        'lr':
        cfg.head_lr,
        'weight_decay':
        cfg.weight_decay
    }]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=cfg.patience, factor=cfg.factor)
    step = 'epoch'

    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        print(f'Epoch: {epoch + 1}')
        model.train()
        train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), 'best.pt')
            print('Saved Best Model!')

        lr_scheduler.step(valid_loss.avg)


if __name__ == '__main__':
    main()
