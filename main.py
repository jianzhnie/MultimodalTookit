import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from multimodal_exp_args import ModelArguments, MultimodalDataTrainingArguments, OurTrainingArguments
from multimodal_transformers.data import TabPreprocessor
from multimodal_transformers.data.multidomal_dataset import TorchTabularTextDataset
from multimodal_transformers.data.text_encoder import get_text_token
from multimodal_transformers.models import MultidomalModel
from multimodal_transformers.models.tabular import TabMlp
from multimodal_transformers.models.text import AutoModelWithText
from multimodal_transformers.utils.util import create_dir_if_not_exists
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, AutoConfig, AutoTokenizer, HfArgumentParser, default_data_collator, get_scheduler, set_seed

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
        datefmt='%Y/%m/%d/ %H:%M:%S',
        handlers=[stream_handler, file_handler])

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
    train_dataset = TorchTabularTextDataset(
        text_encodings=hf_model_text_input,
        tabular_features=X_tab,
        labels=labels,
        label_list=label_list)

    val_dataset = train_dataset
    test_dataset = train_dataset

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

    logger.info('===== Start training')
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

    multimodal_model = MultidomalModel(
        deeptabular=tabmlp,
        deeptext=text_model,
        head_hidden_dims=[128, 64],
        pred_dim=num_labels)

    # DataLoaders creation:
    data_collator = default_data_collator
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=16,
        num_workers=2,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=16)

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

    # Only show the progress bar once on each machine.
    for epoch in range(10):
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
    for i, batch in enumerate(train_loader):
        target = batch['labels']
        # compute output
        output = model(batch)
        loss = criterion(output, target)
        print(loss)
        train_loss += loss.item()
        # measure accuracy and record loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    return train_loss


def validation(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            target = batch['labels']
            # compute output
            output = model(**batch)
            loss = criterion(output, target)
            print(loss)
            epoch_loss += loss.item()
    return epoch_loss


if __name__ == '__main__':
    main()
