'''
Author: jianzhnie
Date: 2021-11-12 14:42:32
LastEditTime: 2021-11-16 12:11:17
LastEditors: jianzhnie
Description:

'''
from typing import Any, Optional

import numpy as np
import torch
from sklearn.utils import Bunch
from torch.utils.data import Dataset


class WideDeepDataset(Dataset):
    r"""
    Defines the Dataset object to load WideDeep data to the model

    Parameters
    ----------
    X_wide: np.ndarray
        wide input
    X_tab: np.ndarray
        deeptabular input
    X_text: np.ndarray
        deeptext input
    X_img: np.ndarray
        deepimage input
    target: np.ndarray
        target array
    transforms: :obj:`MultipleTransforms`
        torchvision Compose object. See models/_multiple_transforms.py
    """

    def __init__(
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        transforms: Optional[Any] = None,
    ):
        super(WideDeepDataset, self).__init__()
        self.X_wide = X_wide
        self.X_tab = X_tab
        self.X_text = X_text
        self.X_img = X_img
        self.transforms = transforms
        if self.transforms:
            self.transforms_names = [
                tr.__class__.__name__ for tr in self.transforms.transforms
            ]
        else:
            self.transforms_names = []
        self.Y = target

    def __getitem__(self, idx: int):  # noqa: C901
        X = Bunch()
        if self.X_wide is not None:
            X.wide = self.X_wide[idx]
        if self.X_tab is not None:
            X.deeptabular = self.X_tab[idx]
        if self.X_text is not None:
            X.deeptext = self.X_text[idx]
        if self.X_img is not None:
            # if an image dataset is used, make sure is in the right format to
            # be ingested by the conv layers
            xdi = self.X_img[idx]
            # if int must be uint8
            if 'int' in str(xdi.dtype) and 'uint8' != str(xdi.dtype):
                xdi = xdi.astype('uint8')
            # if int float must be float32
            if 'float' in str(xdi.dtype) and 'float32' != str(xdi.dtype):
                xdi = xdi.astype('float32')
            # if there are no transforms, or these do not include ToTensor(),
            # then we need to  replicate what Tensor() does -> transpose axis
            # and normalize if necessary
            if not self.transforms or 'ToTensor' not in self.transforms_names:
                if xdi.ndim == 2:
                    xdi = xdi[:, :, None]
                xdi = xdi.transpose(2, 0, 1)
                if 'int' in str(xdi.dtype):
                    xdi = (xdi / xdi.max()).astype('float32')
            # if ToTensor() is included, simply apply transforms
            if 'ToTensor' in self.transforms_names:
                xdi = self.transforms(xdi)
            # else apply transforms on the result of calling torch.tensor on
            # xdi after all the previous manipulation
            elif self.transforms:
                xdi = self.transforms(torch.tensor(xdi))
            # fill the Bunch
            X.deepimage = xdi
        if self.Y is not None:
            y = self.Y[idx]
            return X, y
        else:
            return X

    def __len__(self):
        if self.X_wide is not None:
            return len(self.X_wide)
        if self.X_tab is not None:
            return len(self.X_tab)
        if self.X_text is not None:
            return len(self.X_text)
        if self.X_img is not None:
            return len(self.X_img)


class TorchTabularTextDataset(Dataset):
    """
    :obj:`TorchDataset` wrapper for text dataset with categorical features
    and numerical features

    Parameters:
        encodings (:class:`transformers.BatchEncoding`):
            The output from encode_plus() and batch_encode() methods (tokens, attention_masks, etc) of
            a transformers.PreTrainedTokenizer
        categorical_feats (:class:`numpy.ndarray`, of shape :obj:`(n_examples, categorical feat dim)`, `optional`, defaults to :obj:`None`):
            An array containing the preprocessed categorical features
        numerical_feats (:class:`numpy.ndarray`, of shape :obj:`(n_examples, numerical feat dim)`, `optional`, defaults to :obj:`None`):
            An array containing the preprocessed numerical features
        labels (:class: list` or `numpy.ndarray`, `optional`, defaults to :obj:`None`):
            The labels of the training examples
        class_weights (:class:`numpy.ndarray`, of shape (n_classes),  `optional`, defaults to :obj:`None`):
            Class weights used for cross entropy loss for classification
        df (:class:`pandas.DataFrame`, `optional`, defaults to :obj:`None`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            TabularConfig instance specifying the configs for TabularFeatCombiner

    """

    def __init__(self,
                 text_encodings,
                 tab_feats,
                 labels=None,
                 label_list=None,
                 class_weights=None):
        self.encodings = text_encodings
        self.tab_feats = tab_feats
        self.labels = labels
        self.class_weights = class_weights
        self.label_list = label_list if label_list is not None else [
            i for i in range(len(np.unique(labels)))
        ]

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(
            self.labels[idx]) if self.labels is not None else None
        item['cat_feats'] = torch.tensor(self.tab_feats[idx]).float() \
            if self.tab_feats is not None else torch.zeros(0)
        return item

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        """returns the label names for classification."""
        return self.label_list