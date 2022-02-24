import os
import sys

import numpy as np
import pandas as pd
import torch
from evaluation import build_compute_metrics_fn
from multimodal_exp_args import ModelArguments, MultimodalDataTrainingArguments, OurTrainingArguments
from multimodal_transformers.data import TabPreprocessor
from multimodal_transformers.data.multidomal_dataset import TorchTabularTextDataset
from multimodal_transformers.data.text_encoder import get_text_token
from multimodal_transformers.models import MultidomalModel
from multimodal_transformers.models.tabular import TabMlp
from multimodal_transformers.models.text import AutoModelWithText
from multimodal_transformers.utils.model import data2device
from multimodal_transformers.utils.util import create_dir_if_not_exists
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, Trainer, set_seed

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

    trainer = Trainer(
        model=multimodal_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=build_compute_metrics_fn(task),
    )

    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.
            isdir(model_args.model_name_or_path) else None)
        trainer.save_model()

    # # DataLoaders creation:
    # data_collator = default_data_collator
    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     sampler=train_sampler,
    #     batch_size=16,
    #     num_workers=2,
    # )
    # val_dataloader = DataLoader(val_dataset, batch_size=16)

    # # Optimizer
    # # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {
    #         'params': [
    #             p for n, p in multimodal_model.named_parameters()
    #             if not any(nd in n for nd in no_decay)
    #         ],
    #         'weight_decay':
    #         training_args.weight_decay,
    #     },
    #     {
    #         'params': [
    #             p for n, p in multimodal_model.named_parameters()
    #             if any(nd in n for nd in no_decay)
    #         ],
    #         'weight_decay':
    #         0.0,
    #     },
    # ]
    # optimizer = AdamW(
    #     optimizer_grouped_parameters, lr=training_args.learning_rate)

    # lr_scheduler = get_scheduler(
    #     name=training_args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=50,
    #     num_training_steps=10000,
    # )
    # criterion = nn.CrossEntropyLoss()
    # multimodal_model.to(device)

    # # Only show the progress bar once on each machine.
    # for epoch in range(10):
    #     train(
    #         train_loader=train_dataloader,
    #         model=multimodal_model,
    #         criterion=criterion,
    #         optimizer=optimizer,
    #         lr_scheduler=lr_scheduler)

    #     validation(
    #         val_loader=val_dataloader,
    #         model=multimodal_model,
    #         criterion=criterion)


def train(train_loader, model, criterion, optimizer, lr_scheduler):
    # switch to train mode
    train_loss = 0
    model.train()
    for i, batch in enumerate(train_loader):
        batch = data2device(batch)
        target = batch['labels']
        # compute output
        output = model(batch)
        loss = criterion(output, target)
        print('loss', loss)
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
            output = model(batch)
            loss = criterion(output, target)
            print(loss)
            epoch_loss += loss.item()
    return epoch_loss


if __name__ == '__main__':
    main()
