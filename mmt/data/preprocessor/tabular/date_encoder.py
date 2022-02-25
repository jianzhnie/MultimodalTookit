'''
Author: jianzhnie
Date: 2021-11-12 15:40:06
LastEditTime: 2022-02-24 16:24:51
LastEditors: jianzhnie
Description:

'''

import re
from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

from .base_preprocessor import BasePreprocessor


class DatePreprocessor(BasePreprocessor):

    def __init__(self,
                 date_columns: List[str] = None,
                 encode_date_columns=True):
        super(DatePreprocessor, self).__init__()

        self.date_cols = date_columns
        self.encode_date_columns = encode_date_columns
        self.is_fitted = False

    def preprocess_date(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.encode_date_columns:
            for field_name, freq in self.date_columns:
                data = self.make_date(df, field_name)
                data, added_features = self.add_datepart(
                    data, field_name, frequency=freq, prefix=None, drop=True)

        # The only features that are added are the date features extracted
        # from the date which are categorical in nature
        if added_features is not None:
            self.categorical_cols += added_features
            self.categorical_dim = (
                len(self.categorical_cols)
                if self.categorical_cols is not None else 0)

    # adapted from gluonts
    @classmethod
    def time_features_from_frequency_str(cls, freq_str: str) -> List[str]:
        """Returns a list of time features that will be appropriate for the
        given frequency string.

        Parameters
        ----------

        freq_str
            Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
        """

        features_by_offsets = {
            offsets.YearBegin: [],
            offsets.YearEnd: [],
            offsets.MonthBegin: [
                'Month',
                'Quarter',
                'Is_quarter_end',
                'Is_quarter_start',
                'Is_year_end',
                'Is_year_start',
            ],
            offsets.MonthEnd: [
                'Month',
                'Quarter',
                'Is_quarter_end',
                'Is_quarter_start',
                'Is_year_end',
                'Is_year_start',
            ],
            offsets.Week: [
                'Month',
                'Quarter',
                'Is_quarter_end',
                'Is_quarter_start',
                'Is_year_end',
                'Is_year_start',
                'Is_month_start',
                'Week',
            ],
            offsets.Day: [
                'Month',
                'Quarter',
                'Is_quarter_end',
                'Is_quarter_start',
                'Is_year_end',
                'Is_year_start',
                'Is_month_start',
                'Week'
                'Day',
                'Dayofweek',
                'Dayofyear',
            ],
            offsets.BusinessDay: [
                'Month',
                'Quarter',
                'Is_quarter_end',
                'Is_quarter_start',
                'Is_year_end',
                'Is_year_start',
                'Is_month_start',
                'Week'
                'Day',
                'Dayofweek',
                'Dayofyear',
            ],
            offsets.Hour: [
                'Month',
                'Quarter',
                'Is_quarter_end',
                'Is_quarter_start',
                'Is_year_end',
                'Is_year_start',
                'Is_month_start',
                'Week'
                'Day',
                'Dayofweek',
                'Dayofyear',
                'Hour',
            ],
            offsets.Minute: [
                'Month',
                'Quarter',
                'Is_quarter_end',
                'Is_quarter_start',
                'Is_year_end',
                'Is_year_start',
                'Is_month_start',
                'Week'
                'Day',
                'Dayofweek',
                'Dayofyear',
                'Hour',
                'Minute',
            ],
        }

        offset = to_offset(freq_str)

        for offset_type, feature in features_by_offsets.items():
            if isinstance(offset, offset_type):
                return feature

        supported_freq_msg = f"""
        Unsupported frequency {freq_str}

        The following frequencies are supported:

            Y, YS   - yearly
                alias: A
            M, MS   - monthly
            W   - weekly
            D   - daily
            B   - business days
            H   - hourly
            T   - minutely
                alias: min
        """
        raise RuntimeError(supported_freq_msg)

    # adapted from fastai
    @classmethod
    def make_date(cls, df: pd.DataFrame, date_field: str):
        """Make sure `df[date_field]` is of the right date type."""
        field_dtype = df[date_field].dtype
        if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            field_dtype = np.datetime64
        if not np.issubdtype(field_dtype, np.datetime64):
            df[date_field] = pd.to_datetime(
                df[date_field], infer_datetime_format=True)
        return df

    # adapted from fastai
    @classmethod
    def add_datepart(
        cls,
        df: pd.DataFrame,
        field_name: str,
        frequency: str,
        prefix: str = None,
        drop: bool = True,
    ):
        """Helper function that adds columns relevant to a date in the column
        `field_name` of `df`."""
        field = df[field_name]
        prefix = (re.sub('[Dd]ate$', '', field_name)
                  if prefix is None else prefix) + '_'
        attr = cls.time_features_from_frequency_str(frequency)
        added_features = []
        for n in attr:
            if n == 'Week':
                continue
            df[prefix + n] = getattr(field.dt, n.lower())
            added_features.append(prefix + n)
        # Pandas removed `dt.week` in v1.1.10
        if 'Week' in attr:
            week = (
                field.dt.isocalendar().week
                if hasattr(field.dt, 'isocalendar') else field.dt.week)
            df.insert(3, prefix + 'Week', week)
            added_features.append(prefix + 'Week')
        # TODO Not adding Elapsed by default. Need to route it through config
        # mask = ~field.isna()
        # df[prefix + "Elapsed"] = np.where(
        #     mask, field.values.astype(np.int64) // 10 ** 9, None
        # )
        # added_features.append(prefix + "Elapsed")
        if drop:
            df.drop(field_name, axis=1, inplace=True)

        # Removing features woth zero variations
        for col in added_features:
            if len(df[col].unique()) == 1:
                df.drop(columns=col, inplace=True)
                added_features.remove(col)
        return df, added_features
