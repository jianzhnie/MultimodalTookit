from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

from .base_preprocessor import BasePreprocessor, check_is_fitted
from .deeptabular_utils import LabelEncoder


def embed_sz_rule(n_cat):
    r"""Rule of thumb to pick embedding size corresponding to ``n_cat``. Taken
    from fastai's Tabular API"""
    return min(600, round(1.6 * n_cat**0.56))


class TabPreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the ``deeptabular`` component input dataset

    Parameters
    ----------
    categroical_cols: List, default = None
        List containing the name of the columns that will be represented by
        embeddings or a Tuple with the name and the embedding dimension. e.g.:
        [('education',32), ('relationship',16), ...]
    continuous_cols: List, default = None
        List with the name of the so called continuous cols
    scale: bool, default = True
        Bool indicating whether or not to scale/standarise continuous cols.
        The user should bear in mind that all the ``deeptabular`` components
        available within ``pytorch-widedeep`` they also include the
        possibility of normalising the input continuous features via a
        ``BatchNorm`` or a ``LayerNorm`` layer. See
        :obj:`pytorch_widedeep.models.transformers._embedding_layers`
    auto_embed_dim: bool, default = True
        Boolean indicating whether the embedding dimensions will be
        automatically defined via fastai's rule of thumb':
        :math:`min(600, int(1.6 \times n_{cat}^{0.56}))`
    default_embed_dim: int, default=16
        Dimension for the embeddings used for the ``deeptabular``
        component if the embed_dim is not provided in the ``categroical_cols``
        parameter
    already_standard: List, default = None
        List with the name of the continuous cols that do not need to be
        Standarised. For example, you might have Long and Lat in your
        dataset and might want to encode them somehow (e.g. see the
        ``LatLongScalarEnc`` available in the `autogluon
        <https://github.com/awslabs/autogluon/tree/master/tabular/src/autogluon/tabular>`_
        tabular library) and not standarize them any further
    for_transformer: bool, default = False
        Boolean indicating whether the preprocessed data will be passed to a
        transformer-based model
        (See :obj:`pytorch_widedeep.models.transformers`). If ``True``, the
        param ``categroical_cols`` must just be a list containing the categorical
        columns: e.g.:['education', 'relationship', ...] This is because they
        will all be encoded using embeddings of the same dim.
    with_cls_token: bool, default = False
        Boolean indicating if a `'[CLS]'` token will be added to the dataset
        when using transformer-based models. The final hidden state
        corresponding to this token is used as the aggregated representation
        for classification and regression tasks. If not, the categorical
        (and continuous embeddings if present) will be concatenated before
        being passed to the final MLP.
    shared_embed: bool, default = False
        Boolean indicating if the embeddings will be "shared" when using
        transformer-based models. The idea behind ``shared_embed`` is
        described in the Appendix A in the `TabTransformer paper
        <https://arxiv.org/abs/2012.06678>`_: `'The goal of having column
        embedding is to enable the model to distinguish the classes in one
        column from those in the other columns'`. In other words, the idea is
        to let the model learn which column is embedded at the time. See:
        :obj:`pytorch_widedeep.models.transformers._layers.SharedEmbeddings`.
    verbose: int, default = 1

    Attributes
    ----------
    embed_dim: Dict
        Dictionary where keys are the embed cols and values are the embedding
        dimensions. If ``for_transformer`` is set to ``True`` the embedding
        dimensions are the same for all columns and this attributes is not
        generated during the ``fit`` process
    label_encoder: LabelEncoder
        see :class:`pytorch_widedeep.utils.dense_utils.LabelEncder`
    embeddings_input: List
        List of Tuples with the column name, number of individual values for
        that column and the corresponding embeddings dim, e.g. [
        ('education', 16, 10), ('relationship', 6, 8), ...]
    standardize_cols: List
        List of the columns that will be standarized
    scaler: StandardScaler
        an instance of :class:`sklearn.preprocessing.StandardScaler`
    column_idx: Dict
        Dictionary where keys are column names and values are column indexes.
        This is be neccesary to slice tensors

    Examples
    --------
    >>> import pandas as pd
    >>> from pytorch_widedeep.preprocessing import TabPreprocessor
    >>> df = pd.DataFrame({'color': ['r', 'b', 'g'], 'size': ['s', 'n', 'l'], 'age': [25, 40, 55]})
    >>> categroical_cols = [('color',5), ('size',5)]
    >>> cont_cols = ['age']
    >>> deep_preprocessor = TabPreprocessor(categroical_cols=continuous_cols, continuous_cols=cont_cols)
    >>> X_tab = deep_preprocessor.fit_transform(df)
    >>> deep_preprocessor.embed_dim
    {'color': 5, 'size': 5}
    >>> deep_preprocessor.column_idx
    {'color': 0, 'size': 1, 'age': 2}
    """

    CONTINUOUS_TRANSFORMS = {
        'quantile_uniform': {
            'callable': QuantileTransformer,
            'params': dict(output_distribution='uniform', random_state=42),
        },
        'quantile_normal': {
            'callable': QuantileTransformer,
            'params': dict(output_distribution='normal', random_state=42),
        },
        'box_cox': {
            'callable': PowerTransformer,
            'params': dict(method='box-cox', standardize=True),
        },
        'yeo_johnson': {
            'callable': PowerTransformer,
            'params': dict(method='yeo-johnson', standardize=True),
        },
        'standard_scaler': {
            'callable': StandardScaler,
            'params': dict(with_mean=True, with_std=True),
        },
    }

    def __init__(
        self,
        categroical_cols: List[str] = None,
        continuous_cols: List[str] = None,
        date_cols: List[str] = None,
        category_encoding_type: str = None,
        continuous_transform_method: str = None,
        handle_na: bool = True,
        auto_embed_dim: bool = True,
        default_embed_dim: int = 16,
        for_transformer: bool = False,
        with_cls_token: bool = False,
        shared_embed: bool = False,
    ):
        super(TabPreprocessor, self).__init__()

        self.categroical_cols = categroical_cols
        self.continuous_cols = continuous_cols
        self.date_cols = date_cols
        self.category_encoding_type = category_encoding_type
        self.continuous_transform_method = continuous_transform_method
        self.handle_na = handle_na
        self.auto_embed_dim = auto_embed_dim
        self.default_embed_dim = default_embed_dim
        self.for_transformer = for_transformer
        self.with_cls_token = with_cls_token
        self.shared_embed = shared_embed
        self.is_fitted = False

        if (self.categroical_cols is None) and (self.continuous_cols is None):
            raise ValueError(
                "'categroical_cols' and 'continuous_cols' are 'None'. Please, define at least one of the two."
            )

        transformer_error_message = (
            "If for_transformer is 'True' categroical_cols must be a list "
            ' of strings with the columns to be encoded as embeddings.')
        if self.for_transformer and self.categroical_cols is None:
            raise ValueError(transformer_error_message)
        if self.for_transformer and isinstance(self.categroical_cols[0],
                                               tuple):  # type: ignore[index]
            raise ValueError(transformer_error_message)

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        """Fits the Preprocessor and creates required attributes."""
        if self.categroical_cols is not None:
            df_emb = self._prepare_embed(df)
            self.label_encoder = LabelEncoder(
                columns_to_encode=df_emb.columns.tolist(),
                shared_embed=self.shared_embed,
                for_transformer=self.for_transformer,
            )
            self.label_encoder.fit(df_emb)
            self.embeddings_input: List = []
            for k, v in self.label_encoder.encoding_dict.items():
                if self.for_transformer:
                    self.embeddings_input.append((k, len(v)))
                else:
                    self.embeddings_input.append(
                        (k, len(v), self.embed_dim[k]))

        if (self.continuous_transform_method
                is not None) and (len(self.continuous_cols) > 0):
            transform_method = self.CONTINUOUS_TRANSFORMS[
                self.continuous_transform_method]
            self.continuous_transformer = transform_method['callable'](
                **transform_method['params'])
            df_cont = self._prepare_continuous(df)
            self.continuous_transformer.fit(df_cont)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the processed ``dataframe`` as a np.ndarray."""
        check_is_fitted(self, condition=self.is_fitted)
        if self.categroical_cols is not None:
            df_emb = self._prepare_embed(df)
            df_emb = self.label_encoder.transform(df_emb)
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.continuous_transform_method:
                df_cont[
                    self.
                    continuous_cols] = self.continuous_transformer.transform(
                        df_cont)
        try:
            df_deep = pd.concat([df_emb, df_cont], axis=1)
        except NameError:
            try:
                df_deep = df_emb.copy()
            except NameError:
                df_deep = df_cont.copy()
        self.column_idx = {k: v for v, k in enumerate(df_deep.columns)}
        return df_deep.values

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        r"""Takes as input the output from the ``transform`` method and it will
        return the original values.

        Parameters
        ----------
        encoded: np.ndarray
            array with the output of the ``transform`` method
        """
        decoded = pd.DataFrame(encoded, columns=self.column_idx.keys())
        # embeddings back to original category
        if self.categroical_cols is not None:
            if isinstance(self.categroical_cols[0], tuple):
                emb_c: List = [c[0] for c in self.continuous_cols]
            else:
                emb_c = self.categroical_cols.copy()
            for c in emb_c:
                decoded[c] = decoded[c].map(
                    self.label_encoder.inverse_encoding_dict[c])
        # continuous_cols back to non-standarised
        try:
            decoded[
                self.
                continuous_cols] = self.continuous_transformer.inverse_transform(
                    decoded[self.continuous_cols])
        except AttributeError:
            pass

        if 'cls_token' in decoded.columns:
            decoded.drop('cls_token', axis=1, inplace=True)

        return decoded

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines ``fit`` and ``transform``"""
        return self.fit(df).transform(df)

    def _prepare_embed(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.handle_na:
            for c in self.categroical_cols:
                df.loc[:, c] = df.loc[:, c].astype(str)
            df[self.continuous_cols].fillna(
                df[self.continuous_cols].mode(), inplace=True)

        if self.for_transformer:
            if self.with_cls_token:
                df_cls = df.copy()[self.categroical_cols]
                df_cls.insert(loc=0, column='cls_token', value='[CLS]')
                return df_cls
            else:
                return df.copy()[self.categroical_cols]
        else:
            if self.auto_embed_dim:
                n_cats = {
                    col: df[col].nunique()
                    for col in self.categroical_cols
                }
                self.embed_dim = {
                    col: embed_sz_rule(n_cat)
                    for col, n_cat in n_cats.items()
                }  # type: ignore[misc]
                embed_colname = self.categroical_cols  # type: ignore
            else:
                self.embed_dim = {
                    e: self.default_embed_dim
                    for e in self.categroical_cols
                }  # type: ignore
                embed_colname = self.categroical_cols  # type: ignore
            return df.copy()[embed_colname]

    def _prepare_continuous(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()[self.continuous_cols]
        df[self.continuous_cols] = df[self.continuous_cols].astype(float)
        if self.handle_na:
            df[self.continuous_cols] = df[self.continuous_cols].fillna(
                dict(df[self.continuous_cols].median()), inplace=False)
        return df


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv(
        '/media/robin/DATA/datatsets/structure_data/titanic/Titanic.csv')
    cat_cols = ['Sex', 'Embarked']
    con_cols = ['Fare', 'Age']
    print(df[cat_cols + con_cols])
    tabpreprocessor = TabPreprocessor(
        categroical_cols=cat_cols,
        continuous_cols=con_cols,
        continuous_transform_method='standard_scaler')
    full_data_transformed = tabpreprocessor.fit_transform(df)
    print(full_data_transformed)
    print(tabpreprocessor.embed_dim)
    print(tabpreprocessor.embeddings_input)
    df = tabpreprocessor.inverse_transform(full_data_transformed)
    print(df)
