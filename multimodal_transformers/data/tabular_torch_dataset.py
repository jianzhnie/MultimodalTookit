from functools import partial
import logging
from os.path import join, exists
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data.dataset import Dataset
from .multidomal_dataset import TorchTabularTextDataset
import logging
from functools import partial
import pandas as pd
from .utils import agg_text_columns_func, convert_to_func, get_matching_cols
from .preprocessor import TabPreprocessor

logger = logging.getLogger(__name__)


class MultimodalDatasets(Dataset):
    """
    Function to load tabular and text data from a specified folder into folds

    Loads train, test and/or validation text and tabular data from specified
    csv path into num_splits of train, val and test for Kfold cross validation.
    Performs categorical and numerical data preprocessing if specified. `data_csv_path` is a path to

    Args:
        data_csv_path (str): The path to the csv containing the data
        num_splits (int): The number of cross validation folds to split the data into.
        validation_ratio (float): A float between 0 and 1 representing the percent of the data to hold as a consistent validation set.
        text_cols (:obj:`list` of :obj:`str`): The column names in the dataset that contain text
            from which we want to load
        tokenizer (:obj:`transformers.tokenization_utils.PreTrainedTokenizer`):
            HuggingFace tokenizer used to tokenize the input texts as specifed by text_cols
        label_col (str): The column name of the label, for classification the column should have
            int values from 0 to n_classes-1 as the label for each class.
            For regression the column can have any numerical value
        label_list (:obj:`list` of :obj:`str`, optional): Used for classification;
            the names of the classes indexed by the values in label_col.
        categorical_cols (:obj:`list` of :obj:`str`, optional): The column names in the dataset that
            contain categorical features. The features can be already prepared numerically, or
            could be preprocessed by the method specified by categorical_encode_type
        numerical_cols (:obj:`list` of :obj:`str`, optional): The column names in the dataset that contain numerical features.
            These columns should contain only numeric values.
        sep_text_token_str (str, optional): The string token that is used to separate between the
            different text columns for a given data example. For Bert for example,
            this could be the [SEP] token.
        categorical_encode_type (str, optional): Given categorical_cols, this specifies
            what method we want to preprocess our categorical features.
            choices: [ 'ohe', 'binary', None]
            see encode_features.CategoricalFeatures for more details
        numerical_transformer_method (str, optional): Given numerical_cols, this specifies
            what method we want to use for normalizing our numerical data.
            choices: ['yeo_johnson', 'box_cox', 'quantile_normal', None]
            see https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
            for more details
        empty_text_values (:obj:`list` of :obj:`str`, optional): specifies what texts should be considered as
            missing which would be replaced by replace_empty_text
        replace_empty_text (str, optional): The value of the string that will replace the texts
            that match with those in empty_text_values. If this argument is None then
            the text that match with empty_text_values will be skipped
        max_token_length (int, optional): The token length to pad or truncate to on the
            input text
        debug (bool, optional): Whether or not to load a smaller debug version of the dataset

        Returns:
            :obj:`tuple` of `list` of `tabular_torch_dataset.TorchTextDataset`:
                This tuple contains three lists representing the splits of
                training, validation and testing sets. The length of the lists is
                equal to the number of folds specified by `num_splits`
        """

    def __init__(self,
                 data_csv_path=None,
                 num_splits=None,
                 validation_ratio=None,
                 text_cols=None,
                 tokenizer=None,
                 label_col=None,
                 categorical_cols=None,
                 numerical_cols=None,
                 sep_text_token_str=' ',
                 categorical_encode_type='ohe',
                 numerical_transformer_method='quantile_normal',
                 empty_text_values=None,
                 replace_empty_text=None,
                 max_token_length=None,
                 debug=False):

        self.data_csv_path = data_csv_path
        self.num_splits = num_splits
        self.validation_ratio = validation_ratio
        self.text_cols = text_cols
        self.tokenizer = tokenizer
        self.target = label_col
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.sep_text_token_str = sep_text_token_str
        self.categorical_encode_type = categorical_encode_type
        self.numerical_transformer_method = numerical_transformer_method
        self.empty_text_values = empty_text_values
        self.replace_empty_text = replace_empty_text
        self.max_token_length = max_token_length
        self.debug = debug

    def _image_preprocessor(self):

        pass

    def _text_preprocessor(self,
                           data_df,
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
            texts_list,
            padding=True,
            truncation=True,
            max_length=max_token_length)
        tokenized_text_ex = ' '.join(
            tokenizer.convert_ids_to_tokens(
                hf_model_text_input['input_ids'][0]))
        logger.debug(f'Tokenized text example: {tokenized_text_ex}')

        return hf_model_text_input

    def _tabular_preprocessor(self, df, categroical_cols, continuous_cols,
                              continuous_transform_method):
        tabpreprocessor = TabPreprocessor(
            categroical_cols=categroical_cols,
            continuous_cols=continuous_cols,
            continuous_transform_method=continuous_transform_method)
        data_transformed = tabpreprocessor.fit_transform(df)
        return data_transformed

    def load_data(
        self,
        data_df,
        text_cols,
        tokenizer,
        label_col,
        label_list=None,
        categorical_cols=None,
        numerical_cols=None,
        sep_text_token_str=' ',
        categorical_encode_type='ohe',
        numerical_transformer=None,
        empty_text_values=None,
        replace_empty_text=None,
        max_token_length=None,
        debug=False,
    ):
        """Function to load a single dataset given a pandas DataFrame

        Given a DataFrame, this function loads the data to a :obj:`torch_dataset.TorchTextDataset`
        object which can be used in a :obj:`torch.utils.data.DataLoader`.

        Args:
            data_df (:obj:`pd.DataFrame`): The DataFrame to convert to a TorchTextDataset
            text_cols (:obj:`list` of :obj:`str`): the column names in the dataset that contain text
                from which we want to load
            tokenizer (:obj:`transformers.tokenization_utils.PreTrainedTokenizer`):
                HuggingFace tokenizer used to tokenize the input texts as specifed by text_cols
            label_col (str): The column name of the label, for classification the column should have
                int values from 0 to n_classes-1 as the label for each class.
                For regression the column can have any numerical value
            label_list (:obj:`list` of :obj:`str`, optional): Used for classification;
                the names of the classes indexed by the values in label_col.
            categorical_cols (:obj:`list` of :obj:`str`, optional): The column names in the dataset that
                contain categorical features. The features can be already prepared numerically, or
                could be preprocessed by the method specified by categorical_encode_type
            numerical_cols (:obj:`list` of :obj:`str`, optional): The column names in the dataset that contain numerical features.
                These columns should contain only numeric values.
            sep_text_token_str (str, optional): The string token that is used to separate between the
                different text columns for a given data example. For Bert for example,
                this could be the [SEP] token.
            categorical_encode_type (str, optional): Given categorical_cols, this specifies
                what method we want to preprocess our categorical features.
                choices: [ 'ohe', 'binary', None]
                see encode_features.CategoricalFeatures for more details
            numerical_transformer (:obj:`sklearn.base.TransformerMixin`): The sklearn numeric
                transformer instance to transform our numerical features
            empty_text_values (:obj:`list` of :obj:`str`, optional): Specifies what texts should be considered as
                missing which would be replaced by replace_empty_text
            replace_empty_text (str, optional): The value of the string that will replace the texts
                that match with those in empty_text_values. If this argument is None then
                the text that match with empty_text_values will be skipped
            max_token_length (int, optional): The token length to pad or truncate to on the
                input text
            debug (bool, optional): Whether or not to load a smaller debug version of the dataset

        Returns:
            :obj:`tabular_torch_dataset.TorchTextDataset`: The converted dataset
        """
        if debug:
            data_df = data_df[:500]
        if empty_text_values is None:
            empty_text_values = ['nan', 'None']

        text_cols_func = convert_to_func(text_cols)
        categorical_cols_func = convert_to_func(categorical_cols)
        numerical_cols_func = convert_to_func(numerical_cols)

        categorical_feats, numerical_feats = load_cat_and_num_feats(
            data_df, categorical_cols_func, numerical_cols_func,
            categorical_encode_type)
        numerical_feats = normalize_numerical_feats(numerical_feats,
                                                    numerical_transformer)
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
        labels = data_df[label_col].values

        return TorchTabularTextDataset(hf_model_text_input, categorical_feats,
                                       numerical_feats, labels, data_df,
                                       label_list)
