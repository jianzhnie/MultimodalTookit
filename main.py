import logging
import os
import sys

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from mmt.data.multimodal_dataset import MMDataset
from mmt.data.preprocessor import TabPreprocessor
from mmt.data.utils.text_token import get_text_token
from mmt.models import MultiModalModel
from mmt.models.tabular import TabMlp
from mmt.models.text import AutoModelWithText
from mmt.utils.model import data2device
from mmt.utils.utils import create_dir_if_not_exists, get_args_info_as_str
from multimodal_exp_args import ModelArguments, MultimodalDataTrainingArguments, OurTrainingArguments
from torch import nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, AutoConfig, AutoTokenizer, HfArgumentParser, get_scheduler, set_seed

logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    data_df = pd.read_csv(
        os.path.join(data_args.data_path, 'train.csv'), index_col=0)
    categorical_cols = data_args.column_info['cat_cols']
    numerical_cols = data_args.column_info['num_cols']
    numerical_transformer_method = data_args.numerical_transformer_method

    tab_preprocessor = TabPreprocessor(
        categroical_cols=categorical_cols,
        continuous_cols=numerical_cols,
        continuous_transform_method=numerical_transformer_method)

    X_tab = tab_preprocessor.fit_transform(data_df)

    hf_model_text_input = get_text_token(
        data_df=data_df,
        text_cols=data_args.column_info['text_cols'],
        tokenizer=tokenizer,
        sep_text_token_str=tokenizer.sep_token
        if not data_args.column_info['text_col_sep_token'] else
        data_args.column_info['text_col_sep_token'],
        empty_text_values=None,
        max_token_length=training_args.max_token_length,
    )

    label_col = data_args.column_info['label_col']
    label_list = data_args.column_info['label_list']
    labels = data_df[label_col].values

    train_dataset = MMDataset(
        text_encodings=hf_model_text_input,
        tabular_features=X_tab,
        labels=labels,
        label_list=label_list)

    val_dataset = train_dataset

    # model
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

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    text_model = AutoModelWithText.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir)

    multimodal_model = MultiModalModel(
        deeptabular=tabmlp,
        deeptext=text_model,
        head_hidden_dims=[128, 64],
        pred_dim=num_labels)

    # DataLoaders creation:
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=8, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in multimodal_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            training_args.weight_decay,
        },
        {
            'params': [
                p for n, p in multimodal_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=training_args.learning_rate)

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=10000,
    )

    criterion = nn.CrossEntropyLoss()
    multimodal_model.to(device)

    # Only show the progress bar once on each machine.
    for epoch in range(training_args.num_train_epochs):
        multimodal_model.train()
        train(
            train_loader=train_dataloader,
            model=multimodal_model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler)

        validation(
            val_loader=val_dataloader,
            model=multimodal_model,
            criterion=criterion)


def train(train_loader, model, criterion, optimizer, lr_scheduler):
    # switch to train mode
    train_loss = 0
    model.train()
    for step, batch in enumerate(train_loader):
        batch = data2device(batch, device=device)
        target = batch['labels']
        # compute output
        output = model(batch)
        loss = criterion(output, target)
        print('loss', loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    return train_loss


def validation(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            batch = data2device(batch, device=device)
            target = batch['labels']
            # compute output
            output = model(batch)
            loss = criterion(output, target)
            epoch_loss += loss.item()
    return epoch_loss


if __name__ == '__main__':
    main()
