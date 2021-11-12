'''
Author: jianzhnie
Date: 2021-11-12 20:28:07
LastEditTime: 2021-11-12 21:03:11
LastEditors: jianzhnie
Description: 

'''

from functools import partial
import logging
from os.path import join, exists
from utils import agg_text_columns_func, get_matching_cols, convert_to_func
import pandas as pd
logger = logging.getLogger(__name__)


class TextEncoder(object):

    def __init__(self):
        """
        Args:
        text_cols (:obj:`list` of :obj:`str`): the column names in the dataset that contain text
                from which we want to load
        tokenizer (:obj:`transformers.tokenization_utils.PreTrainedTokenizer`):
            HuggingFace tokenizer used to tokenize the input texts as specifed by text_cols
        sep_text_token_str (str, optional): The string token that is used to separate between the
            different text columns for a given data example. For Bert for example,
            this could be the [SEP] token.
        empty_text_values (:obj:`list` of :obj:`str`, optional): Specifies what texts should be considered as
            missing which would be replaced by replace_empty_text
        replace_empty_text (str, optional): The value of the string that will replace the texts
            that match with those in empty_text_values. If this argument is None then
            the text that match with empty_text_values will be skipped
        max_token_length (int, optional): The token length to pad or truncate to on the
            input text
        debug (bool, optional): Whether or not to load a smaller debug version of the dataset  
        """
        pass

    def transform(self,
                  data_df,
                  text_cols,
                  tokenizer,
                  sep_text_token_str=' ',
                  empty_text_values=None,
                  replace_empty_text=None,
                  max_token_length=None):

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
            texts_list,
            padding=True,
            truncation=True,
            max_length=max_token_length)
        tokenized_text_ex = ' '.join(
            tokenizer.convert_ids_to_tokens(
                hf_model_text_input['input_ids'][0]))
        logger.debug(f'Tokenized text example: {tokenized_text_ex}')

        return hf_model_text_input, data_df


if __name__ == '__main__':
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    df = pd.read_csv(
        "/home/robin/jianzh/multimodal/Multimodal-Toolkit/datasets/Womens_Clothing_E-Commerce_Reviews/test.csv"
    )
    text_encoder = TextEncoder().transform(
        df, text_cols=["Title", "Review Text"], tokenizer=tokenizer)
    print(text_encoder)
