"""Dataset implementation for specific task(s)"""
import logging
import math
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger()


def _absolute_pathify(df, root=None, column='image'):
    """Convert relative paths to absolute."""
    if root is None:
        return df
    assert column in df.columns
    assert isinstance(root, str), 'Invalid root path: {}'.format(root)
    for i, _ in df.iterrows():
        path = df.at[i, column]
        if not os.path.isabs(path):
            df.at[i, column] = os.path.join(root, os.path.expanduser(path))
    return df


def img_from_csv(df, root=None, image_column='image'):
    r"""Create from csv file.

    Parameters
    ----------
    csv_file : str
        The path for csv file.
    root : str
        The relative root for image paths stored in csv file.
    image_column : str, default is 'image'
        The name of the column for image paths.
    """
    assert image_column in df.columns, f'`{image_column}` column is required, used for accessing the original images'
    df = _absolute_pathify(df, root=root, column=image_column)
    df = df.rename(
        columns={
            image_column: 'image',
        }, errors='ignore')
    df = df.reset_index(drop=True)
    return df


class TorchImageDataset(Dataset):
    """Internal wrapper read entries in pd.DataFrame as images/labels.

    Parameters
    ----------
    dataset : ImageClassificationDataset
        DataFrame as ImageClassificationDataset.
    transform : Torchvision Transform function
        torch function for image transformation
    """

    def __init__(self,
                 dataset,
                 input_size=224,
                 crop_ratio=0.875,
                 is_train=True):
        assert isinstance(dataset, pd.DataFrame)
        assert 'image' in dataset.columns
        self._dataset = dataset
        self._imread = Image.open

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        jitter_param = 0.4
        crop_ratio = crop_ratio if crop_ratio > 0 else 0.875
        resize = int(math.ceil(input_size / crop_ratio))
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=jitter_param,
                contrast=jitter_param,
                saturation=jitter_param),
            transforms.ToTensor(), normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(), normalize
        ])

        if is_train is None:
            self.transform = transform_train
        else:
            self.transform = transform_test

    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, idx):
        im_path = self._dataset['image'][idx]
        img = self._imread(im_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


if __name__ == '__main__':
    csv_file = '/Users/jianzhengnie/work/MultimodalTransformers/a.csv'
    df = pd.read_csv(csv_file)
    df = img_from_csv(df, root='root', image_column='image_name')
    print(df)
