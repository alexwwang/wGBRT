# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:34:43 2022

@author: alex
@email: alex@chuanxilu.com

Copyright (C) 2022 alex

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

from typing import List, Tuple, Union, Optional
# from random import shuffle

import pandas as pd
import numpy as np
from numpy.typing import NDArray
# from sklearn.utils import shuffle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

def column_transform(train_set: "DataFrame",
        test_set: "DataFrame" = None,
        valid_set: "DataFrame" = None,
        transformers: List[Tuple] = None,
        **kwargs
        ) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Preprocess dataset, including scaling, shuffling and
    transforming.

    Parameters
    ----------
    train_set: DataFrame
        Training dataset.
    test_set: DataFrame, optional
        Testing dataset. The default is None.
    valid_set: DataFrame, optional
        Validating dataset. The default is None.
    transformers: List[Tuple]
        List of (name, transformer, columns) tuples specifying the transformer
        objects to be applied to subsets of the data. The default is None.
    kwargs: dict
        Keyword args for ColumnTransformer.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        Datasets preprocessed. Input n datasets, then output n datasets.
    """
    if transformers is None:
        return train_set.values, test_set.values, valid_set.values
    column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder=MinMaxScaler(),
            sparse_threshold=kwargs.get("sparse_threshold", 0.3),
            n_jobs=kwargs.get("n_jobs", None),
            transformer_weights=kwargs.get("transformer_weights", None),
            verbose=kwargs.get("verbose", False),
            verbose_feature_names_out=kwargs.get("verbose_feature_names_out",
                                                True))
    train_transformed = column_transformer.fit_transform(train_set)
    test_transformed = column_transformer.transform(test_set) if \
            test_set is not None else test_set
    valid_transformed = column_transformer.transform(valid_set) if \
            valid_set is not None else valid_set
    return train_transformed, test_transformed, valid_transformed


def processing(data: NDArray, processor: object) -> NDArray:
    """
    Processing data with `processor` in given features.
    A processor could be an Imputater, Scaler or Transformer
    instance.

    Parameters
    ----------
    data: NDArray
        Features data to process.
    processor: object
        Object instance initialized with `*_initializer`.

    Returns
    -------
    NDArray:
        Features data processed by processor.

    """
    try:
        transformed = processor.transform(data)
    except AttributeError as ae:
        raise ae("`transform` method not found in processor.")
    return transformed


def scaler_initializer(data: "DataFrame",
        scaler: Union[str, object], **kwargs) -> object:
    """
    Initialize a Scaler by training data.

    Parameters
    ----------
    data: DataFrame
        Training data to initialize a scaler.
    scaler: Union[str, object]
        Using scaler from sklearn or customized object
        that has `fit` and `transform` method.
        Support `MinMaxScaler`, `StandardScaler`, `MaxAbsScaler`
        and `RobustScaler` from `sklearn`.
    kwargs:
        Keyword args for scaler.
        Warning: should not pass other args into or error occures.

    Returns
    -------
    object
        A Scaler object that has `transform` method.
    """
    if isinstance(scaler, str):
        if scaler in ["MinMaxScaler", "StandardScaler",
                "MaxAbsScaler", "RobustScaler"]:
            try:
                exec(f"from sklearn.preprocessing import {scaler}")
            except ModuleNotFoundError as mnfe:
                raise mnfe(f"sklearn or {scaler} not found.")
        else:
            raise ModuleNotFoundError(f"{scaler} not found.")
        _scaler = eval(scaler)(**kwargs).fit(data)
        return _scaler
    try:
        _scaler = scaler(**kwargs).fit(data)
    except AttributeError as ae:
        raise ae("`fit` method not found in scaler.")
    except NameError as ne:
        raise ne("`scaler` is not defined.")
    except TypeError as te:
        raise te("`scaler` object is not callable.")
    return _scaler


def imputater_initializer(data: "DataFrame",
        imputater: str, **kwargs) -> object:
    """
    Initialize an imputater that fills missing value in `data`
    with given `imputater` from `sklearn.impute`.

    Parameters
    ----------
    data: DataFrame
        Features data to imputate missing values.
    imputater: str
        Support imputater names from `sklearn.impute`, e.g.
        `SimpleImputer`, `IterativeImputer`, `KNNImputer`.
    kwargs:
        Keyword args for imputer.
        Warning: should not pass other args into or error occures.

    Returns
    -------
    object:
        An imputater initialized and fited with train_data.

    """
    if not all([isinstance(imputater, str),
        imputater in ["SimpleImputer", "IterativeImputer", "KNNImputer"]]):
        raise ModuleNotFoundError(f"{imputater} not found.")
    else:
        try:
            if imputater == "IterativeImputer":
                exec("from sklearn.experimental import enable_iterative_imputer")
            exec(f"from sklearn.impute import {imputater}")
        except ModuleNotFoundError as mnfe:
            raise mnfe(f"sklearn module not found.")
    _imputater = eval(imputater)(**kwargs)
    _imputater.fit(data)
    return _imputater


def _sin(x: NDArray, period: int) -> NDArray:
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    return np.sin(x / period * 2 * np.pi)


def _cos(x: NDArray, period: int) -> NDArray:
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    return np.cos(x / period * 2 * np.pi)


def sin_transformer(period: int) -> FunctionTransformer:
    return FunctionTransformer(func=_sin, kw_args={"period": period},
            validate=True)


def cos_transformer(period: int) -> FunctionTransformer:
    return FunctionTransformer(func=_cos, kw_args={"period": period},
            validate=True)


def transformer_initializer(data: "DataFrame",
        transformer: Union[str, object], **kwargs) -> object:
    """
    Initialize a Transformer by training data.

    Parameters
    ----------
    train_data: DataFrame
        Training data to initialize a transformer.
    transformer: Union[str, object]
        Using transformer from sklearn or customized object
        that has `fit` and `transform` method.
        Support `FunctionTransformer`, `PowerTransformer`,
        `QuantileTransformer` and `SplineTransformer` from `sklearn`.
    kwargs:
        Keyword args for transformer.
        Warning: should not pass other args into or error occures.

    Returns
    -------
    object
        A transformer object that has `transform` method.
    """
    if isinstance(transformer, str):
        if transformer in ["FunctionTransformer", "PowerTransformer",
                "QuantileTransformer", "SplineTransformer"]:
            try:
                exec(f"from sklearn.preprocessing import {transformer}")
            except ModuleNotFoundError as mnfe:
                raise mnfe(f"sklearn or {transformer} not found.")
        else:
            raise ModuleNotFoundError(f"{transformer} not found.")
        _transformer = eval(transformer)(**kwargs).fit(data)
        return _transformer
    try:
        _transformer = transformer(**kwargs).fit(data)
    except AttributeError as ae:
        raise ae("`fit` method not found in transformer.")
    except NameError as ne:
        raise ne("`transformer` is not defined.")
    except TypeError as te:
        raise te("`transformer` object is not callable.")
    return _transformer


def make_rolling(ts_data: NDArray,
        periods_in: int,
        periods_out: int,
        channel_labels: int = 1,
        ) -> Tuple[NDArray, NDArray]:
    """
    A transformer processes time series data.
    Rolling along `ts_data`, a 2-D dataframe period by period with a
    certain length determined by input periods plusing output periods.

    Parameters
    ----------
    ts_data: NDArray
        2-D dataframe time series with label on the 1st `channel_labels`
        column(s) and other columns as features. The sum number of columns
        is `ts_data.shape[1] = channel_labels + num_feats`.
    periods_in: int
        Input period length.
    periods_out: int
        Output period length.
    channel_labels: int
        Channel number of labels in `ts_data`. The default is 1.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Rolling Frames as a Tuple(Features, Labels)
        Features dim: [num_batches, periods_in, channel_labels + num_feats]
        Labels dim: [num_batches, periods_out, channel_labels]
        If `channel_labels == 1` then Labels dim:
        [num_batches, periods_in + num_feats]
    """
    num_feats = ts_data.shape[1] - channel_labels
    if num_feats < 0:
        raise ValueError("`num_feats` < 0, `channel_labels` does not match " +
                "with dims of `ts_data`.")
    stop = ts_data.shape[0]
    start = 0
    window_len = periods_in + periods_out
    feats = []
    labels = []
    # ts_data = ts_data.values
    for pointer in range(start, stop - window_len, 1):
        input_rb = pointer + periods_in
        output_rb = input_rb + periods_out
        feats.append(ts_data[pointer: input_rb, :])
        labels.append(ts_data[input_rb: output_rb, 0: channel_labels])
    # change feats list into an array
    feats = np.asarray(feats)
    # reshape the array into dim:
    # [num_batches, periods_in, channel_labels + num_feats]
    feats = feats.reshape(-1, periods_in, num_feats + channel_labels)
    # change labels list into an array
    labels = np.asarray(labels)
    # reshape the array into dim: [num_batches, periods_out, channel_labels]
    labels = labels.reshape(-1, periods_out, channel_labels)
    if channel_labels == 1:
        labels = labels.squeeze(axis=2)
    return feats, labels


def shuffle_batch(feats: NDArray,
        labels: NDArray,
        **kwargs) -> Tuple[NDArray, NDArray]:
    """
    Shuffle batches of feats and labels simutaniously.

    Parameters
    ----------
    feats: NDArray
        Features array, dim:
        [num_batches, periods_in, channel_labels + num_feats]
    labels: NDArray
        Labels array, dim: [num_batches, periods_out, channel_labels]
    kwargs:
        Keyword args for `shuffle` from sklearn.utils.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Shuffled rolling frames as Tuple(Features, Labels)
        Features dim: [num_batches, periods_in, channel_labels + num_feats]
        Labels dim: [num_batches, periods_out, channel_labels]

    """
    from sklearn.utils import shuffle
    feats, labels = shuffle(feats, labels,
            random_state=kwargs.get("random_state", None),
            n_samples=kwargs.get("n_samples", None))
    return feats, labels


def build_wgbrt_feats(feats: NDArray,
        channel_labels: int) -> NDArray:
    """
    Build time series features data structure fitting for GBRT model.

    Parameters
    ----------
    feats: NDArray
        Features array, dim:
        [num_batches, periods_in, channel_labels + num_feats]
    channel_labels: int
        Channel numbers of labels.

    Returns
    -------
    NDArray
        Features array, dim:
        [num_batches, periods_in + num_feats, channel_labels]
        If `channel_labels == 1` then dim:
        [num_batches, periods_in + num_feats]
    """
    num_batches, periods_in, total_cols = feats.shape
    num_feats = total_cols - channel_labels
    if num_feats < 0:
        raise ValueError("`num_feats` < 0, `channel_labels` does not match " +
                "with dims of `ts_data`.")
    # direct matrix transform implement
    # slice a label sub-matrix with dim[num_batches, periods_in, channel_labels]
    # from `feats` matrix, then transpose last two axis to get
    # `period_labels` dim: [num_batches, channel_labels, periods_in]
    period_labels = feats[:, :, :channel_labels].transpose([0, 2, 1])
    # slice a feat sub-matrix with dim[num_batches, 1, num_feats] as
    # `recent_feats` & automatically squeezed to dim: [num_batches, num_feats]
    recent_feats = feats[:, -1, -num_feats:]
    # expand `recent_feats` dim back to [num_batches, 1, num_feats], so we can
    # repeat it on axis 1 to get dim [num_batches, channel_labels, num_feats]
    recent_feats_repeat = np.expand_dims(recent_feats, axis=1
                            ).repeat(channel_labels, axis=1)
    # concatenate `period_labels` with `recent_feats_repeat` on axis 2,
    # then transpose axis 1 and axis 2 to get
    # `new_feats` dim: [num_batches, period_in + num_feats, channel_labels]
    new_feats = np.concatenate([period_labels, recent_feats_repeat], axis=2
                            ).transpose([0, 2, 1])
    # squeeze on `channel_labels` dimension, to get a 2D feats frame
    if channel_labels == 1:
        new_feats = new_feats.squeeze(axis=2)
    return new_feats

