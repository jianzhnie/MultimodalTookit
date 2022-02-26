"""Dataset implementation for specific task(s)"""
# pylint: disable=consider-using-generator
import logging
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger()


def _absolute_pathify(df, root=None, column='image'):
    """Convert relative paths to absolute."""
    if root is None:
        return df
    assert column in df.columns
    assert isinstance(root, str), 'Invalid root path: {}'.format(root)
    root = os.path.abspath(os.path.expanduser(root))
    for i, _ in df.iterrows():
        path = df.at[i, 'image']
        if not os.path.isabs(path):
            df.at[i, 'image'] = os.path.join(root, os.path.expanduser(path))
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
    label_column : str, default is 'label'
        The name for the label column, leave it as is if no label column is available. Note that
        in such case you won't be able to train with this dataset, but can still visualize the images.
    """
    assert image_column in df.columns, f'`{image_column}` column is required, used for accessing the original images'
    df = _absolute_pathify(df, root=root, column=image_column)
    return df


class TorchImageClassificationDataset(Dataset):
    """Internal wrapper read entries in pd.DataFrame as images/labels.

    Parameters
    ----------
    dataset : ImageClassificationDataset
        DataFrame as ImageClassificationDataset.
    transform : Torchvision Transform function
        torch function for image transformation
    """

    def __init__(self, dataset, transform=None):
        assert isinstance(dataset, pd.DataFrame)
        assert 'image' in dataset.columns
        self._dataset = dataset
        self._imread = Image.open
        self.transform = transform

    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, idx):
        im_path = self._dataset['image'][idx]
        img = self._imread(im_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


if __name__ == '__main__':
    import autogluon.core as ag
    csv_file = ag.utils.download(
        'https://autogluon.s3-us-west-2.amazonaws.com/datasets/petfinder_example.csv'
    )
    df = pd.read_csv(csv_file)
    df = img_from_csv(df, root='root')
    print(df)
