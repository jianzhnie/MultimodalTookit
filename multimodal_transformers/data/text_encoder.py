'''
Author: jianzhnie
Date: 2021-11-12 20:28:07
LastEditTime: 2021-11-16 17:41:56
LastEditors: jianzhnie
Description:

'''

import logging
from functools import partial

import pandas as pd

from .utils import agg_text_columns_func, convert_to_func, get_matching_cols

logger = logging.getLogger(__name__)


def get_text_token(data_df,
                   text_cols,
                   tokenizer,
                   sep_text_token_str=' ',
                   empty_text_values=None,
                   replace_empty_text=None,
                   max_token_length=None):

    if empty_text_values is None:
        empty_text_values = ['nan', 'None']
    text_cols_func = convert_to_func(text_cols)

    agg_func = partial(agg_text_columns_func, empty_text_values,
                       replace_empty_text)
    texts_cols = get_matching_cols(data_df, text_cols_func)
    logger.info(f'Text columns: {texts_cols}')
    texts_list = data_df[texts_cols].agg(agg_func, axis=1).tolist()
    for i, text in enumerate(texts_list):
        texts_list[i] = f' {sep_text_token_str} '.join(text)
    logger.info(f'Raw text example: {texts_list[0]}')
    hf_model_text_input = tokenizer(
        texts_list, padding=True, truncation=True, max_length=max_token_length)
    tokenized_text_ex = ' '.join(
        tokenizer.convert_ids_to_tokens(hf_model_text_input['input_ids'][0]))
    logger.debug(f'Tokenized text example: {tokenized_text_ex}')

    return hf_model_text_input


if __name__ == '__main__':
    import torch
    from torch import tensor
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    df = pd.read_csv(
        '/home/robin/jianzh/multimodal/Multimodal-Toolkit/datasets/Womens_Clothing_E-Commerce_Reviews/test.csv'
    )
    text_cols = ['Title', 'Review Text']
    text_cols = ['Division Name', 'Department Name', 'Class Name']
    print(df[text_cols])

    text_cols_func = convert_to_func(text_cols)
    empty_text_values = ['nan', 'None']
    replace_empty_text = None
    agg_func = partial(agg_text_columns_func, empty_text_values,
                       replace_empty_text)
    print(agg_func)
    texts_cols = get_matching_cols(df, text_cols_func)
    print(text_cols)
    logger.info(f'Text columns: {texts_cols}')
    texts_list = df[texts_cols].agg(agg_func, axis=1).tolist()
    text_encoder, df = get_text_token(
        df,
        text_cols=['Title', 'Review Text'],
        tokenizer=tokenizer,
        sep_text_token_str=tokenizer.sep_token,
        max_token_length=16,
    )
    item = {key: torch.tensor(val[0]) for key, val in text_encoder.items()}
    print(item)

    item = {
        'input_ids':
        tensor([
            101, 2307, 2801, 1010, 3532, 7781, 102, 1045, 7078, 3866, 1996,
            2801, 1997, 2019, 17876, 102
        ]),
        'token_type_ids':
        tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'attention_mask':
        tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    }
