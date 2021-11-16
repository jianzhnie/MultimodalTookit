'''
Author: jianzhnie
Date: 2021-11-12 15:40:06
LastEditTime: 2021-11-16 14:37:48
LastEditors: jianzhnie
Description:

'''

import logging
from typing import List

import numpy as np
import pandas as pd
from preprocessor.base_preprocessor import BasePreprocessor, check_is_fitted
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

logger = logging.getLogger(__name__)


class NumericalFeatureTransformer(BasePreprocessor):
    """
    CONTINUOUS_TRANSFORMS = {
        "quantile_uniform": {
            "callable": QuantileTransformer,
            "params": dict(output_distribution="uniform", random_state=42),
        },
        "quantile_normal": {
            "callable": QuantileTransformer,
            "params": dict(output_distribution="normal", random_state=42),
        },
        "box_cox": {
            "callable": PowerTransformer,
            "params": dict(method="box-cox", standardize=True),
        },
        "yeo_johnson": {
            "callable": PowerTransformer,
            "params": dict(method="yeo-johnson", standardize=True),
        },
        "nomalize": {
            "callable": StandardScaler,
            "params": dict(with_mean=True, with_std=True),
        }
    }
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
        'nomalize': {
            'callable': StandardScaler,
            'params': dict(with_mean=True, with_std=True),
        }
    }

    def __init__(self,
                 numerical_cols: List[str] = None,
                 numerical_transformer_method: str = None,
                 handle_na: bool = True):
        super(NumericalFeatureTransformer, self).__init__()

        self.numerical_cols = numerical_cols
        self.numerical_transformer_method = numerical_transformer_method
        self.handle_na = handle_na
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        if (self.numerical_transformer_method
                is not None) and (len(self.numerical_cols) > 0):
            transform = self.CONTINUOUS_TRANSFORMS[
                self.numerical_transformer_method]
            self.numerical_transformer = transform['callable'](
                **transform['params'])
            df_cont = self._prepare_continuous(df)
            self.transformer = self.numerical_transformer.fit(df_cont)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the processed ``dataframe`` as a np.ndarray."""
        check_is_fitted(self, condition=self.is_fitted)
        if self.numerical_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.numerical_transformer_method:
                df_cont[self.numerical_cols] = self.transformer.transform(
                    df_cont)

        self.column_idx = {k: v for v, k in enumerate(df_cont.columns)}
        return df_cont.values

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        r"""Takes as input the output from the ``transform`` method and it will
        return the original values.

        Parameters
        ----------
        encoded: np.ndarray
            array with the output of the ``transform`` method
        """
        decoded = pd.DataFrame(encoded, columns=self.column_idx.keys())
        try:
            decoded[self.numerical_cols] = self.transformer.inverse_transform(
                decoded[self.numerical_cols])
        except AttributeError:
            pass
        return decoded

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines ``fit`` and ``transform``"""
        return self.fit(df).transform(df)

    def _prepare_continuous(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()[self.numerical_cols]
        df[self.numerical_cols] = df[self.numerical_cols].astype(float)
        if self.handle_na:
            df[self.numerical_cols] = df[self.numerical_cols].fillna(
                dict(df[self.numerical_cols].median()), inplace=False)
        return df


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv(
        '/media/robin/DATA/datatsets/structure_data/titanic/Titanic.csv')
    cols = ['Fare', 'Age']
    print(df[cols])
    cat_feats = NumericalFeatureTransformer(
        numerical_cols=cols, numerical_transformer_method='quantile_uniform')
    full_data_transformed = cat_feats.fit_transform(df)
    print(full_data_transformed)
    df = cat_feats.inverse_transform(full_data_transformed)
    print(df)
