import logging
from typing import List, Optional

import pandas as pd
from multimodal_transformers.data import TabPreprocessor
from multimodal_transformers.data.multidomal_dataset import TabularImageTextDataset
from multimodal_transformers.data.text_encoder import text_token
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)


class MultiFeildDatasets(Dataset):

    def __init__(self,
                 data_csv_path: str = None,
                 num_splits: int = None,
                 validation_ratio: float = None,
                 text_cols: List[str] = None,
                 tokenizer: Optional[List] = None,
                 categorical_cols=None,
                 numerical_cols=None,
                 sep_text_token_str=' ',
                 categorical_encode_type='ohe',
                 numerical_transformer_method='quantile_normal',
                 empty_text_values=None,
                 replace_empty_text=None,
                 max_token_length=None,
                 debug=False) -> None:

        self.data_csv_path = data_csv_path
        self.num_splits = num_splits
        self.validattion_ratio = validation_ratio
        self.text_cols = text_cols
        self.tokenizer = tokenizer
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.sep_text_token_str = sep_text_token_str
        self.categorical_encode_type = categorical_encode_type
        self.numerical_transformer_method = numerical_transformer_method
        self.empty_text_values = empty_text_values
        self.replace_empty_text = replace_empty_text
        self.max_token_length = max_token_length

        all_data_df = pd.read_csv(data_csv_path)
        train_df, val_df = train_test_split(
            all_data_df,
            test_size=validation_ratio,
            shuffle=True,
            train_size=1 - validation_ratio,
            random_state=5)

        dfs = [df for df in [train_df, val_df] if df is not None]
        data_df = pd.concat(dfs, axis=0)
        tab_processor = TabPreprocessor(
            categorical_cols=categorical_cols,
            continuous_cols=numerical_cols,
            category_encoding_type=categorical_encode_type,
            continuous_transform_method=numerical_transformer_method)

        vals = tab_processor.fit_transform()
        data_df = pd.DataFrame(vals, columns=tab_processor.feat_names)

        len_train = len(train_df)
        len_val = len(val_df) if val_df is not None else 0

        train_df = data_df.iloc[:len_train]
        if val_df is not None:
            val_df = data_df.iloc[len_train:len_train + len_val]
            len_train = len_train + len_val

        hf_model_text_input, df = text_token(
            data_df=data_df,
            text_cols=text_cols,
            sep_text_token_str=sep_text_token_str,
            empty_text_values=empty_text_values,
            replace_empty_text=replace_empty_text,
            max_token_length=max_token_length)

        categorical_feats = data_df[categorical_cols]
        numerical_feats = data_df[numerical_cols]

        return TabularImageTextDataset(
            text_encodings=hf_model_text_input,
            categorical_feats=categorical_feats,
            numerical_feats=numerical_feats)
