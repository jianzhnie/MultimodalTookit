import logging
import os
import sys
from pprint import pformat
from statistics import mean, stdev
from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch
from evaluation import calc_classification_metrics, calc_regression_metrics
from multimodal_exp_args import ModelArguments, MultimodalDataTrainingArguments, OurTrainingArguments
from multimodal_transformers.data import TabPreprocessor
from multimodal_transformers.data.multidomal_dataset import TorchTabularTextDataset
from multimodal_transformers.data.text_encoder import text_token
from multimodal_transformers.models.config import TabularConfig
from multimodal_transformers.models.tabular import TabMlp
from multimodal_transformers.models.text import AutoModelWithText
from multimodal_transformers.utils.util import create_dir_if_not_exists, get_args_info_as_str
from scipy.special import softmax
from transformers import AutoConfig, AutoTokenizer, EvalPrediction, HfArgumentParser, Trainer, set_seed

os.environ['COMET_MODE'] = 'DISABLED'
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, MultimodalDataTrainingArguments,
                               OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        )

    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.'
        )

    # Setup logging
    create_dir_if_not_exists(training_args.output_dir)
    stream_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(
        filename=os.path.join(training_args.output_dir, 'train_log.txt'),
        mode='w+')
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        level=logging.INFO
        if training_args.local_rank in [-1, 0] else logging.WARN,
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[stream_handler, file_handler])

    logger.info(
        f'======== Model Args ========\n{get_args_info_as_str(model_args)}\n')
    logger.info(
        f'======== Data Args ========\n{get_args_info_as_str(data_args)}\n')
    logger.info(
        f'======== Training Args ========\n{get_args_info_as_str(training_args)}\n'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    data_df = pd.read_csv(data_args.data_path)
    categorical_cols = data_args.column_info['cat_cols']
    numerical_cols = data_args.column_info['num_cols'],
    numerical_transformer_method = data_args.numerical_transformer_method

    tab_preprocessor = TabPreprocessor(
        categroical_cols=categorical_cols,
        continuous_cols=numerical_cols,
        continuous_transform_method=numerical_transformer_method)

    X_tab = tab_preprocessor.fit_transform(data_df)

    hf_model_text_input = text_token(
        data_df=data_df,
        text_cos=data_args.column_info['text_cols'],
        tokenizer=tokenizer,
        sep_text_token_str=tokenizer.sep_token
        if not data_args.column_info['text_col_sep_token'] else
        data_args.column_info['text_col_sep_token'],
        empty_text_values=None,
        max_token_length=training_args.max_token_length,
    )

    label_col = data_args.column_info['label_col'],
    label_list = data_args.column_info['label_list'],
    labels = data_df[label_col].values
    train_dataset = TorchTabularTextDataset(
        text_encodings=hf_model_text_input,
        tab_feats=X_tab,
        labels=labels,
        label_list=label_list)

    train_datasets = [train_dataset]
    val_datasets = [train_dataset]
    test_datasets = [train_dataset]

    tabmlp = TabMlp(
        mlp_hidden_dims=[8, 4],
        column_idx=tab_preprocessor.column_idx,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=tab_preprocessor.continuous_cols)

    set_seed(training_args.seed)
    task = data_args.task
    if task == 'regression':
        num_labels = 1
    else:
        num_labels = len(
            np.unique(train_dataset.labels)
        ) if data_args.num_classes == -1 else data_args.num_classes

    def build_compute_metrics_fn(
            task_name: str) -> Callable[[EvalPrediction], Dict]:

        def compute_metrics_fn(p: EvalPrediction):
            if task_name == 'classification':
                preds_labels = np.argmax(p.predictions, axis=1)
                if p.predictions.shape[-1] == 2:
                    pred_scores = softmax(p.predictions, axis=1)[:, 1]
                else:
                    pred_scores = softmax(p.predictions, axis=1)
                return calc_classification_metrics(pred_scores, preds_labels,
                                                   p.label_ids)
            elif task_name == 'regression':
                preds = np.squeeze(p.predictions)
                return calc_regression_metrics(preds, p.label_ids)
            else:
                return {}

        return compute_metrics_fn


if __name__ == '__main__':
    main()
