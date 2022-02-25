'''
Author: jianzhnie
Date: 2021-11-12 14:42:32
LastEditTime: 2022-02-25 11:30:12
LastEditors: jianzhnie
Description:

'''

import numpy as np
import torch
from torch.utils.data import Dataset


class MMDataset(Dataset):
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
                 tabular_features,
                 image_features=None,
                 labels=None,
                 label_list=None,
                 class_weights=None):
        self.encodings = text_encodings
        self.tabular_features = tabular_features
        self.image_features = image_features
        self.labels = labels
        self.class_weights = class_weights
        self.label_list = label_list if label_list is not None else [
            i for i in range(len(np.unique(labels)))
        ]

    def __getitem__(self, idx):
        item = dict()
        item['deeptext'] = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item['deeptabular'] = torch.tensor(self.tabular_features[idx]).float() \
            if self.tabular_features is not None else torch.zeros(0)
        item['deepimage'] = torch.tensor(self.image_features[idx]).float() \
            if self.image_features is not None else torch.zeros(0)
        item['labels'] = torch.tensor(
            self.labels[idx]) if self.labels is not None else None
        return item

    # def __getitem__(self, idx):
    #     item = dict()
    #     item = {
    #         key: torch.tensor(val[idx])
    #         for key, val in self.encodings.items()
    #     }
    #     item['deeptabular'] = torch.tensor(self.tabular_features[idx]).float() \
    #         if self.tabular_features is not None else torch.zeros(0)
    #     item['deepimage'] = torch.tensor(self.image_features[idx]).float() \
    #         if self.image_features is not None else torch.zeros(0)
    #     item['labels'] = torch.tensor(
    #         self.labels[idx]) if self.labels is not None else None
    #     return item

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        """returns the label names for classification."""
        return self.label_list
