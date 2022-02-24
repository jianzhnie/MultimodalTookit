import logging
import os
import sys

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from multimodal_exp_args import ModelArguments, MultimodalDataTrainingArguments, OurTrainingArguments
from multimodal_transformers.data import TabPreprocessor
from multimodal_transformers.data.multidomal_dataset import TorchTabularTextDataset
from multimodal_transformers.data.text_encoder import get_text_token
from multimodal_transformers.models import MultidomalModel
from multimodal_transformers.models.tabular import TabMlp
from multimodal_transformers.models.text import AutoModelWithText
from multimodal_transformers.utils.model import data2device
from multimodal_transformers.utils.util import create_dir_if_not_exists
from torch import nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, AutoConfig, AutoTokenizer, HfArgumentParser, default_data_collator, get_scheduler, set_seed

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

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        +
        f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    create_dir_if_not_exists(training_args.output_dir)

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
        collate_fn=data_collator,
        sampler=train_sampler,
        batch_size=16,
        num_workers=2,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset, collate_fn=data_collator, batch_size=16)

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

    # Prepare everything with our `accelerator`.
    multimodal_model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        multimodal_model, optimizer, train_dataloader, val_dataloader)

    criterion = nn.CrossEntropyLoss()
    multimodal_model.to(device)

    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {training_args.num_train_epochs}')
    logger.info(
        f'  Instantaneous batch size per device = {training_args.per_device_train_batch_size}'
    )
    logger.info(
        f'  Total train batch size (w. parallel, distributed & accumulation) = {training_args.total_batch_size}'
    )
    logger.info(
        f'  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}'
    )
    logger.info(
        f'  Total optimization steps = {training_args.max_train_steps}')

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


def train(train_loader, model, criterion, accelerator, optimizer, lr_scheduler,
          args):
    # switch to train mode
    train_loss = 0
    model.train()
    for step, batch in enumerate(train_loader):
        batch = data2device(batch)
        target = batch['labels']
        # compute output
        output = model(**batch)
        loss = criterion(output, target)
        loss = loss / args.gradient_accumulation_steps
        accelerator.backward(loss)
        if step % args.gradient_accumulation_steps == 0 or step == len(
                train_loader) - 1:
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
            batch = data2device(batch)
            target = batch['labels']
            # compute output
            output = model(batch)
            loss = criterion(output, target)
            epoch_loss += loss.item()
    return epoch_loss


if __name__ == '__main__':
    main()
