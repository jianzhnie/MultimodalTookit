'''
Author: jianzhnie
Date: 2021-11-09 14:40:19
LastEditTime: 2021-11-11 16:29:58
LastEditors: jianzhnie
Description:

'''
from collections import OrderedDict

import torch.nn as nn
from transformers import (AlbertConfig, AlbertForSequenceClassification,
                          AutoConfig, AutoModelForSequenceClassification,
                          BertConfig, BertForSequenceClassification,
                          DistilBertConfig,
                          DistilBertForSequenceClassification, RobertaConfig,
                          RobertaForSequenceClassification, XLMConfig,
                          XLMForSequenceClassification, XLNetConfig,
                          XLNetForSequenceClassification)
from transformers.configuration_utils import PretrainedConfig

MODEL_FOR_SEQUENCE_WITH_TABULAR_CLASSIFICATION_MAPPING = OrderedDict([
    (
        RobertaConfig,
        RobertaForSequenceClassification,
    ),
    (BertConfig, BertForSequenceClassification),
    (DistilBertConfig, DistilBertForSequenceClassification),
    (AlbertConfig, AlbertForSequenceClassification),
    (XLNetConfig, XLNetForSequenceClassification),
    (XLMConfig, XLMForSequenceClassification),
])


class AutoTextModel(nn.Module):

    def __init__(self, args) -> None:
        self.args = args

        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.text_model, num_labels=args.num_classes)

    def forward(self, X):
        out = self.model(X)
        return out


class BertWithTabular(BertForSequenceClassification):
    """Bert Model transformer with a sequence classification/regression head as
    well as a TabularFeatCombiner module to combine categorical and numerical
    features with the Bert pooled output.

    Parameters:
        hf_model_config (:class:`~transformers.BertConfig`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            :obj:`TabularConfig` instance specifying the configs for :obj:`TabularFeatCombiner`
    """

    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                class_weights=None,
                output_attentions=None,
                output_hidden_states=None,
                cat_feats=None,
                numerical_feats=None):
        r"""
        class_weights (:obj:`torch.FloatTensor` of shape :obj:`(tabular_config.num_labels,)`, `optional`, defaults to :obj:`None`):
            Class weights to be used for cross entropy loss function for classification task
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`tabular_config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`tabular_config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        cat_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.cat_feat_dim)`, `optional`, defaults to :obj:`None`):
            Categorical features to be passed in to the TabularFeatCombiner
        numerical_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.numerical_feat_dim)`, `optional`, defaults to :obj:`None`):
            Numerical features to be passed in to the TabularFeatCombiner
    Returns:
        :obj:`tuple` comprising various elements depending on configuration and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if tabular_config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.num_labels)`):
            Classification (or regression if tabular_config.num_labels==1) scores (before SoftMax).
        classifier_layer_outputs(:obj:`list` of :obj:`torch.FloatTensor`):
            The outputs of each layer of the final classification layers. The 0th index of this list is the
            combining module's output
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        return pooled_output


class AutoModelWithText:

    def __init__(self):
        raise EnvironmentError(
            'AutoModelWithTabular is designed to be instantiated '
            'using the `AutoModelWithTabular.from_pretrained(pretrained_model_name_or_path)` or '
            '`AutoModelWithTabular.from_config(config)` methods.')

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Only the models in multimodal_transformers.py are implemented

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:
                    see multimodal_transformers.py for supported transformer models

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelWithTabular.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_FOR_SEQUENCE_WITH_TABULAR_CLASSIFICATION_MAPPING.items(
        ):
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            'Unrecognized configuration class {} for this kind of AutoModel: {}.\n'
            'Model type should be one of {}.'.format(
                config.__class__,
                cls.__name__,
                ', '.join(
                    c.__name__ for c in
                    MODEL_FOR_SEQUENCE_WITH_TABULAR_CLASSIFICATION_MAPPING.
                    keys()),
            ))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        r""" Instantiates one of the sequence classification model classes of the library
        from a pre-trained model configuration.
        See multimodal_transformers.py for supported transformer models

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaining positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = AutoModelWithTabular.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelWithTabular.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelWithTabular.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop('config', None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                                **kwargs)

        for config_class, model_class in MODEL_FOR_SEQUENCE_WITH_TABULAR_CLASSIFICATION_MAPPING.items(
        ):
            if isinstance(config, config_class):
                return model_class.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    config=config,
                    **kwargs)
        raise ValueError(
            'Unrecognized configuration class {} for this kind of AutoModel: {}.\n'
            'Model type should be one of {}.'.format(
                config.__class__,
                cls.__name__,
                ', '.join(
                    c.__name__ for c in
                    MODEL_FOR_SEQUENCE_WITH_TABULAR_CLASSIFICATION_MAPPING.
                    keys()),
            ))


if __name__ == '__main__':
    model_args = {
        'output_dir': './logs_petfinder/',
        'debug_dataset': True,
        'task': 'classification',
        'num_labels': 5,
        'combine_feat_method': 'text_only',
        'experiment_name': 'bert-base-multilingual-uncased',
        'model_name_or_path': 'bert-base-multilingual-uncased',
        'do_train': True,
        'tokenizer_name': 'bert-base-multilingual-uncased',
        'per_device_train_batch_size': 12,
        'gpu_num': 0,
        'num_train_epochs': 5,
        'categorical_encode_type': 'ohe',
        'use_class_weights': True,
        'logging_steps': 50,
        'eval_steps': 750,
        'save_steps': 3000,
        'learning_rate': 1e-4,
        'data_path': './datasets/PetFindermy_Adoption_Prediction/',
        'column_info_path':
        './datasets/PetFindermy_Adoption_Prediction/column_info_all_text.json',
        'config_name': None,
        'overwrite_output_dir': True
    }

    config = AutoConfig.from_pretrained('bert-base-multilingual-uncased_')
    print(config)
    model = AutoModelWithText.from_pretrained(
        'bert-base-multilingual-uncased_')
    print(model)
