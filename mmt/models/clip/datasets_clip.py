'''
Author: jianzhnie
Date: 2021-12-03 11:45:04
LastEditTime: 2021-12-03 16:15:00
LastEditors: jianzhnie
Description:

'''
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch

from .config_clip import ClipConfig as cfg


class CLIPDataset(torch.utils.data.Dataset):

    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """image_filenames and cpations must have the same length; so, if there
        are multiple captions for each image, the image_filenames must have
        repetitive file names."""

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=cfg.max_length)
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f'{cfg.image_path}/{self.image_filenames[idx]}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)


def get_transforms(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.Resize(cfg.size, cfg.size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ])
    else:
        return A.Compose([
            A.Resize(cfg.size, cfg.size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ])


def make_train_valid_dfs():
    dataframe = pd.read_csv(f'{cfg.captions_path}/captions.csv')
    max_id = dataframe['id'].max() + 1
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False)
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe['id'].isin(train_ids)].reset_index(
        drop=True)
    valid_dataframe = dataframe[dataframe['id'].isin(valid_ids)].reset_index(
        drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe['image'].values,
        dataframe['caption'].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True if mode == 'train' else False,
    )
    return dataloader
