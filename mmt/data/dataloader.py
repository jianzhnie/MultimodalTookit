import math

import pandas as pd
import torch
from mmt.data.preprocessor import TabPreprocessor
from mmt.data.preprocessor.vision.image_utils import img_from_csv
from mmt.data.utils.text_token import get_text_token
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MMDataset(Dataset):

    def __init__(self,
                 data_df,
                 text_cols,
                 tokenizer,
                 sep_text_token_str=' ',
                 empty_text_values=None,
                 replace_empty_text=None,
                 max_token_length=None,
                 label_col=None,
                 label_list=None,
                 categorical_cols=None,
                 numerical_cols=None,
                 categorical_encode_type='ohe',
                 numerical_transformer=None,
                 image_col=None,
                 image_path=None,
                 image_input_size=224,
                 crop_ratio=0.875,
                 is_train=True,
                 debug=False):

        if debug:
            data_df = data_df[:500]
        if empty_text_values is None:
            empty_text_values = ['nan', 'None']

        if text_cols is not None:
            self.hf_model_text_input = get_text_token(
                data_df=data_df,
                text_cols=text_cols,
                tokenizer=tokenizer,
                sep_text_token_str=sep_text_token_str,
                empty_text_values=empty_text_values,
                max_token_length=max_token_length,
            )

        if not (categorical_cols is None and numerical_cols is None):
            tab_preprocessor = TabPreprocessor(
                categroical_cols=categorical_cols,
                continuous_cols=numerical_cols,
                category_encoding_type=categorical_encode_type,
                continuous_transform_method=numerical_transformer)

            self.tabular_feats = tab_preprocessor.fit_transform(data_df)

        if image_col is not None:
            image_df = img_from_csv(
                data_df, image_column=image_col, root=image_path)

            assert isinstance(image_df, pd.DataFrame)
            assert 'image' in image_df.columns
            self._dataset = image_df
            self._imread = Image.open

        label_col = label_col
        label_list = label_list
        self.labels = data_df[label_col].values

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        jitter_param = 0.4
        crop_ratio = crop_ratio if crop_ratio > 0 else 0.875
        resize = int(math.ceil(image_input_size / crop_ratio))
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=jitter_param,
                contrast=jitter_param,
                saturation=jitter_param),
            transforms.ToTensor(), normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(image_input_size),
            transforms.ToTensor(), normalize
        ])

        if is_train is None:
            self.transform = transform_train
        else:
            self.transform = transform_test

    def __getitem__(self, idx):
        if self.image_col is not None:
            im_path = self._dataset['image'][idx]
            img = self._imread(im_path).convert('RGB')
            if self.transform is not None:
                img_tensor = self.transform(img)
        else:
            img_tensor = None

        item = dict()
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.hf_model_text_input.items()
        }
        item['deeptabular'] = torch.tensor(self.tabular_features[idx]).float() \
            if self.tabular_features is not None else torch.zeros(0)
        item[
            'deepimage'] = img_tensor if self.img_tensor is not None else torch.zeros(
                0)
        item['labels'] = torch.tensor(
            self.labels[idx]) if self.labels is not None else None
        return item

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        """returns the label names for classification."""
        return self.label_list


def load_dataset(
    data_df,
    text_cols,
    tokenizer,
    sep_text_token_str=' ',
    empty_text_values=None,
    replace_empty_text=None,
    max_token_length=None,
    label_col=None,
    label_list=None,
    categorical_cols=None,
    numerical_cols=None,
    categorical_encode_type='ohe',
    numerical_transformer=None,
    image_col=None,
    image_path=None,
    image_input_size=224,
    crop_ratio=0.875,
    is_train=True,
    debug=False,
) -> None:

    if debug:
        data_df = data_df[:500]
    if empty_text_values is None:
        empty_text_values = ['nan', 'None']

    hf_model_text_input = get_text_token(
        data_df=data_df,
        text_cols=text_cols,
        tokenizer=tokenizer,
        sep_text_token_str=sep_text_token_str,
        empty_text_values=empty_text_values,
        max_token_length=max_token_length,
    )

    tab_preprocessor = TabPreprocessor(
        categroical_cols=categorical_cols,
        continuous_cols=numerical_cols,
        category_encoding_type=categorical_encode_type,
        continuous_transform_method=numerical_transformer)

    tabular_feats = tab_preprocessor.fit_transform(data_df)

    image_df = img_from_csv(data_df, image_column=image_col, root=image_path)

    label_col = label_col
    label_list = label_list
    labels = data_df[label_col].values

    return MMDataset(
        text_encodings=hf_model_text_input,
        tabular_features=tabular_feats,
        image_df=image_df,
        labels=labels,
        label_list=label_list)
