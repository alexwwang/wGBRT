#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:17:23 2022

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

from typing import List, Tuple, Dict, Sequence, Mapping, Union, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import RegressorChain, MultiOutputRegressor
import lightgbm as lgb
import optuna

from feat_transform import make_rolling, shuffle_batch, build_wgbrt_feats


class wLGBM(object):
    def __init__(self,
            data_set: "pd.DataFrame",
            labels: List[str],
            cat_feat_names: List[str],
            is_ts: bool,
            period_in: int = 7,
            period_out: int = 3,
            model_name: str = "wLGBM1",
            feat_prefix: str = "lbl_feat",
            multiout_mode : str = "chain",
            use_multiout: bool = False,
            model_param: dict = None,
            data_param: dict = None,
            train_param: dict = None,
            print_info: bool = True):
        self.model_name = model_name
        self.feat_prefix = feat_prefix
        self.data = data_set
        self.labels = data_set[labels]
        self.channel_labels = len(labels)
        self.feat_nums = self.data.shape[1] - self.channel_labels
        colnames = list(self.data.columns)
        self.label_names = colnames[:self.channel_labels]
        self.feat_names = colnames[-self.feat_nums:]
        self.colnames = [*self.label_names, *self.feat_names]
        self.data = self.data[self.colnames]
        self.transformed_feat_names = None
        self.cat_feat_names = cat_feat_names
        self.is_ts = is_ts
        self.period_in, self.period_out = period_in, period_out
        self.ts_data_splits = None
        self.training_data, self.test_data, self.valid_data = None, None, None
        self.X_train, self.Y_train = None, None
        self.X_valid, self.Y_valid = None, None
        self.X_test, self.Y_test = None, None
        self.dtrain, self.dtest, self.dvalid = None, None, None
        self.core_model, self.model = None, None
        self.multiout_mode = multiout_mode
        self.data_param = data_param
        self.model_param = model_param
        self.train_param = train_param
        self.use_multiout = use_multiout
        self.objective = None
        self.print_info = print_info


    def data_split(self,
            test_rate: float = 0.2,
            valid_rate: float = 0,
            **kwargs):
        data_size = self.data.shape[0]
        test_size = int(np.floor(data_size * test_rate))
        train_size = data_size - test_size
        valid_size = int(np.floor(train_size * valid_rate))
        if self.is_ts:
            tscv = TimeSeriesSplit(
                    n_splits=kwargs.get("n_splits", 5),
                    max_train_size=kwargs.get("max_train_size", None),
                    test_size=test_size,
                    gap=kwargs.get("gap", 0))
            self.ts_data_splits = list(tscv.split(self.data))
            train_idx, test_idx = self.ts_data_splits[-1]
            self.training_data = self.data.iloc[train_idx, :]
            self.test_data = self.data.iloc[test_idx, :]
            self.valid_data = self.training_data.iloc[-valid_size:, :]
        else:
            train, test = train_test_split(self.data,
                    test_size=test_rate,
                    train_size=kwargs.get("train_size", None),
                    random_state=kwargs.get("random_state", None),
                    shuffle=kwargs.get("shuffle", True),
                    stratify=kwargs.get("stratify", None))
            train, valid = train_test_split(train,
                    test_size=valid_rate,
                    train_size=kwargs.get("train_size", None),
                    random_state=kwargs.get("random_state", None),
                    shuffle=kwargs.get("shuffle", True),
                    stratify=kwargs.get("stratify", None))
            self.training_data = train
            self.test_data = test
            self.valid_data = valid


    def preprocess_data(self,
            tasks: Dict[str, List[str]],
            processors: Dict[str, object],
            kw_args: Dict[str, dict],
            **kwargs):
        transformers = []
        for task, cols in tasks.items():
            tasker_kwargs = kw_args.get(task)
            if isinstance(tasker_kwargs, dict):
                tasker = processors.get(task)(**tasker_kwargs)
            else:
                tasker = processors.get(task)
            transformers.append((task, tasker, cols))
        label_not_exclude = [label for label in self.label_names
                             if label not in tasks.get("passthrough", [])]
        transformers.append(("passthrough", "passthrough", label_not_exclude))
        column_transformer = ColumnTransformer(
                transformers=transformers,
                remainder=MinMaxScaler(),
                sparse_threshold=kwargs.get("sparse_threshold", 0.3),
                n_jobs=kwargs.get("n_jobs", None),
                transformer_weights=kwargs.get("transformer_weights", None),
                verbose=kwargs.get("verbose", False),
                verbose_feature_names_out=kwargs.get("verbose_feature_names_out",
                                                 True)
                ).fit(self.training_data)
        self.transformed_feat_names = self._build_transformed_feat_names(
                column_transformer)
        self.training_data = column_transformer.transform(
                self.training_data)
        self.valid_data = column_transformer.transform(
                self.valid_data)
        self.test_data = column_transformer.transform(
                self.test_data)
        label_index = [self.transformed_feat_names.index(label)
                       for label in self.label_names]
        feats_index = [self.transformed_feat_names.index(feat)
                       for feat in self.transformed_feat_names
                       if feat not in self.label_names]
        permutation = [*label_index, *feats_index]
        self.transformed_feat_names = [self.transformed_feat_names[i]
                                       for i in permutation]
        self.ori_cat_feat_names = self.cat_feat_names[:]
        self.cat_feat_names = [col for col in self.transformed_feat_names
                               if col.endswith("_categorical")]
        new_idx = np.asarray(permutation)
        self.training_data[:] = self.training_data[:, new_idx]
        self.valid_data[:] = self.valid_data[:, new_idx]
        self.test_data[:] = self.test_data[:, new_idx]
        if self.print_info:
            print(f"categorical features renamed:")
            print(f"from {self.ori_cat_feat_names} to {self.cat_feat_names}\n")
            print(f"features transformed, training data samples:")
            train_df = pd.DataFrame(self.training_data,
                                    columns=self.transformed_feat_names)
            print(f"{train_df.head()}\n")
        pass


    def _build_transformed_feat_names(self,
            col_transformer: ColumnTransformer) -> list:
        new_feat_names = []
        for label, transformer, cols in col_transformer.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                new_feat_names.extend(transformer.get_feature_names_out())
            elif label.lower() in ["remainder"]:
                new_feat_names.extend(self.colnames[cols])
            elif label.lower() in ["passthrough"]:
                new_feat_names.extend(cols)
            else:
                new_feat_names.extend([f"{col}_{label}" for col in cols])
        if self.print_info:
            print(f"transformed feature names:\n{new_feat_names}\n")
        return new_feat_names


    def _build_wide_feat_names(self, period_in: int):
        self.wfeat_names = [f"{self.feat_prefix}_{i}" for i in
                range(period_in - 1, -1, -1)]
        if self.transformed_feat_names is not None:
            self.wfeat_names.extend(
                self.transformed_feat_names[len(self.label_names):])
        else:
            self.wfeat_names.extend(self.feat_names)
        # print(f"wide feature names built up: \n{self.wfeat_names}")


    def prepare_wide_feats(self,
            period_in: int = None,
            period_out: int = None,
            **kwargs):
        period_in = period_in if period_in is not None else self.period_in
        period_out = period_out if period_out is not None else self.period_out
        wide_feats, multi_labels = zip(*[make_rolling(
                data,
                period_in,
                period_out,
                self.channel_labels) for data in [self.training_data,
                    self.valid_data, self.test_data]
                ])
        self.X_train, self.X_valid, self.X_test = wide_feats
        self.Y_train, self.Y_valid, self.Y_test = multi_labels

        self.X_train, self.Y_train = shuffle_batch(
                                        self.X_train,
                                        self.Y_train,
                                        **kwargs)
        self.X_train = build_wgbrt_feats(self.X_train, self.channel_labels)
        self.X_valid = build_wgbrt_feats(self.X_valid, self.channel_labels)
        self.X_test = build_wgbrt_feats(self.X_test, self.channel_labels)
        self._build_wide_feat_names(period_in)
        if self.print_info:
            print(f"wide features transformed, training data samples:")
            train_df = pd.DataFrame(self.X_train,
                                    columns=self.wfeat_names)
            print(f"{train_df.head()}")
            print(f"label data samples:\n{self.Y_train[:5, :]}")
            print(f"training data shape:\n{self.X_train.shape}")
            print(f"label data shape:\n{self.Y_train.shape}")

    def build_lgbm_data(self, data_param: dict = None):
        data_param = data_param if data_param is not None else self.data_param
        self.dtrain = lgb.Dataset(self.X_train, label=self.Y_train,
                        categorical_feature=self.cat_feat_names,
                        feature_name=self.wfeat_names,
                        params=self.data_param,
                        free_raw_data=True)
        self.dtest = lgb.Dataset(self.X_test, label=self.Y_test,
                        categorical_feature=self.cat_feat_names,
                        feature_name=self.wfeat_names,
                        reference=self.dtrain,
                        params=self.data_param,
                        free_raw_data=True)
        self.dvalid = lgb.Dataset(self.X_valid, label=self.Y_valid,
                        categorical_feature=self.cat_feat_names,
                        feature_name=self.wfeat_names,
                        reference=self.dtrain,
                        params=self.data_param,
                        free_raw_data=True)
        pass


    def build_core_model(self,
                         model_param: dict = None,
                         data_param: dict = None
                         ):
        model_param = model_param if model_param is not None \
                else self.model_param
        data_param = data_param if data_param is not None \
            else self.data_param
        if data_param is not None:
            model_param.update(data_param)
        if model_param is not None:
            self.core_model = lgb.LGBMRegressor(**model_param)
        else:
            raise ValueError("Failed to build a core model. " +
                    "`model_param` is None.")
        pass


    def build_model(self,
            core_model: lgb.LGBMRegressor = None,
            multiout_mode: str = "chain",
            **kwargs):
        core_model = core_model if core_model is not None \
                else self.core_model
        if core_model is None:
            if self.core_model is None:
                if self.model_param is not None:
                    self.build_core_model()
                else:
                    raise ValueError("`core_model` is not initialized. " +
                        "Use `build_core_model` to initialized one.")
        # gbrt_unit = lgb.LGBMRegressor(**model_param)
        multiout_mode = multiout_mode if multiout_mode is not None \
                else self.multiout_mode
        print(f"Use {multiout_mode} mode to build up multioutput model.")
        if multiout_mode == "chain":
            exec("from sklearn.multioutput import RegressorChain")
            self.model = RegressorChain(base_estimator=core_model,
                        order=list(range(self.period_out)),
                        cv=kwargs.get("cv", None),
                        random_state=kwargs.get("random_state", None))
        elif multiout_mode == "standalone":
            exec("from sklearn.multioutput import MultiOutputRegressor")
            self.model = MultiOutputRegressor(estimator=core_model,
                        n_jobs=kwargs.get("n_jobs", None))
        else:
            self.multiout_mode = None
            raise ValueError("`multiout_mode` should only be `chain`" +\
                    "or `standalone`.")
        pass


    def train_core_model(self, train_param: dict = None):
        train_param = train_param if train_param is not None \
                else self.train_param
        Y_valid = self.Y_valid[:, 0]
        Y_train = self.Y_train[:, 0]
        if self.print_info:
            print(f"Y_valid data samples:\n{Y_valid[:5]}")
        train_param.update({"eval_set": [(self.X_valid, Y_valid)]})
        if self.core_model is None:
            self.build_core_model()
        self.core_model.fit(
                self.X_train,
                Y_train,
                feature_name=self.wfeat_names,
                categorical_feature=self.cat_feat_names,
                **train_param)


    def train_model(self, train_param: dict = None):
        train_param = train_param if train_param is not None \
                else self.train_param
        # train_param.update({"eval_set": [(self.X_valid, self.Y_valid)]})
        if self.model is None:
            if self.core_model is not None:
                self.build_model()
            else:
                self.build_core_model()
                self.build_model()
        # self.model.set_params(**train_param)
        self.model.fit(
                self.X_train,
                self.Y_train,
                feature_name=self.wfeat_names,
                categorical_feature=self.cat_feat_names,
                **train_param)
        pass


    def predict(self, x_predict: np.ndarray = None,
                use_multiout: bool = None) -> np.ndarray:
        use_multiout = self.use_multiout if use_multiout is None else use_multiout
        x_predict = x_predict if x_predict is not None else self.X_test
        if use_multiout:
            return self.model.predict(x_predict)
        else:
            return self.core_model.predict(x_predict)


    def metrics(self,
                preds: np.ndarray,
                truth: np.ndarray = None,
                methods: Tuple[Union[str, callable]] = ("mean_squared_error",),
                is_core_model: bool = False
                ) -> Dict[str, float]:
        if is_core_model:
            truth = truth if truth is not None else self.Y_test[:, 0]
        else:
            truth = truth if truth is not None else self.Y_test
        scores = {}
        for method in methods:
            if isinstance(method, str):
                exec(f"from sklearn.metrics import {method}")
                scores.update({method: eval(method)(preds, truth)})
            elif hasattr(method, "__call__") and hasattr(method, "__name__"):
                scores.update({method.__name__: method(preds, truth)})
        if self.print_info:
            print(f"metric scores:\n {scores}")
        return scores


    @staticmethod
    def get_trial_params(trial: optuna.Trial,
                        base_param: dict = None,
                        trial_param: Dict[str, Tuple[str, Sequence]] = None,
                        depend_param: Dict[str, Tuple[str, str, Sequence[Mapping]]] = None,
                        condition_param: Dict[str, Tuple[str, str, str, Sequence]] = None,
                        *args, **kwargs) -> dict:
        depend_param = {} if depend_param is None else depend_param
        condition_param = {} if condition_param is None else condition_param
        trial_param = {} if trial_param is None else trial_param

        _trial_params = {}
        if base_param is not None:
            _trial_params.update(base_param)
        for param_name, settings in trial_param.items():
            try:
                param_type, param_choice = settings
            except TypeError as te:
                print(f"@{param_name}:")
                raise te
            if not isinstance(param_choice, Sequence):
                raise TypeError("`param_choice` in `param_set` " +
                        "should be `Sequence` type.")
            if param_type == "categorical":
                _trial_params.update(
                        {param_name:
                            trial.suggest_categorical(param_name,
                                                    list(param_choice))
                        })
            elif param_type == "int":
                _trial_params.update(
                        {param_name:
                            trial.suggest_int(param_name,
                                            param_choice[0],
                                            param_choice[-1])
                        })
            elif param_type == "float":
                _trial_params.update(
                        {param_name:
                            trial.suggest_float(param_name,
                                            param_choice[0],
                                            param_choice[-1])
                        })

        for param_name, settings in depend_param.items():
            try:
                param_type, depend_name, param_choice = settings
            except TypeError as te:
                print(f"@{param_name}:")
                raise te
            if not isinstance(param_choice, Sequence):
                raise TypeError("`param_choice` in `depend_param` " +
                        "should be `Sequence` type.")
            if param_type == "int":
                _trial_params.update({
                    param_name: trial.suggest_int(
                        param_name,
                        param_choice[0](_trial_params.get(depend_name)),
                        param_choice[-1](_trial_params.get(depend_name))
                    )
                })
            elif param_type == "float":
                _trial_params.update({
                    param_name: trial.suggest_int(
                        param_name,
                        param_choice[0](_trial_params.get(depend_name)),
                        param_choice[-1](_trial_params.get(depend_name))
                    )
                })

        for param_name, settings in condition_param.items():
            try:
                param_type, condition_key, condition_value, param_choice = settings
            except TypeError as te:
                print(f"@{param_name}:")
                raise te
            if not isinstance(param_choice, Sequence):
                raise TypeError("`param_choice` in `condition_param` " +
                        "should be `Sequence` type.")
            if param_type == "int":
                if _trial_params.get(condition_key) == condition_value:
                    _trial_params.update({
                        param_name: trial.suggest_int(
                            param_name,
                            param_choice[0],
                            param_choice[-1]
                        )
                    })
            elif param_type == "float":
                if _trial_params.get(condition_key) == condition_value:
                    _trial_params.update({
                        param_name: trial.suggest_float(
                            param_name,
                            param_choice[0],
                            param_choice[-1]
                        )
                    })

        return _trial_params


    def initial_optimizer(self,
            model_params_sets: Optional[Tuple[Dict]] = (),
            data_params_sets: Optional[Tuple[Dict]] = (),
            train_params_sets: Optional[Tuple[Dict]] = (),
            task_params_sets: Optional[Tuple[Dict]] = (),
            only_train_core: bool = False,
            metric: Union[str, callable] = "mean_squared_error") -> callable:
        def objective(trial: optuna.Trial):
            nonlocal self
            model_params = wLGBM.get_trial_params(trial, *model_params_sets)
            data_params = wLGBM.get_trial_params(trial, *data_params_sets)
            train_params = wLGBM.get_trial_params(trial, *train_params_sets)
            task_params = wLGBM.get_trial_params(trial, *task_params_sets)
            if "period_in" in task_params.keys():
                self.prepare_wide_feats(period_in=task_params.get("period_in"),
                        period_out=task_params.get("period_out"))
            self.build_core_model(model_params, data_params)
            if only_train_core:
                self.train_core_model(train_param=train_params)
                self.use_multiout = False
            else:
                self.build_model(
                        multiout_mode=task_params.get(
                            "multiout_mode", "standalone"
                        )
                )
                self.train_model(train_param=train_params)
                self.use_multiout = True
            scores = self.metrics(
                self.predict(),
                methods=(metric, ),
                is_core_model=only_train_core
            )
            if hasattr(metric, "__call__") and hasattr(metric, "__name__"):
                # print(f"scores: {scores.get(metric.__name__)}")
                return scores.get(metric.__name__)
            # print(f"scores: {scores.get(metric)}")
            return scores.get(metric)

        self.objective = objective
        return objective
        # raise NotImplementedError("This method has not been implemented.")


    def __call__(self, trial: optuna.Trial):
        if self.objective is None:
            raise NotImplementedError("Call `hp_optimizer` first " +\
                    "to implement `objective` func.")
        return self.objective(trial)


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    from sklearn.preprocessing import MinMaxScaler
    from feat_transform import sin_transformer, cos_transformer
    bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2,
            as_frame=True)
    data = bike_sharing.frame
    columns = list(data.columns)
    label_col = columns[-1]
    feats_col = columns[:-1]
    adj_cols = [label_col, *feats_col]
    data = data[adj_cols]
    cat_feat_cols = ["weather", "season", "holiday", "workingday"]
    categories = [
        ["clear", "misty", "rain"],
        ["spring", "summer", "fall", "winter"],
        ["False", "True"],
        ["False", "True"],
    ]
    data["weather"].replace(to_replace="heavy_rain", value="rain",
        inplace=True)
    ordinal_encoder = OrdinalEncoder(categories=categories)
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    # ordinal_encoder.fit(data[cat_feat_cols])
    # data[cat_feat_cols] = ordinal_encoder.transform(data[cat_feat_cols])

    model_param = {
        # Core Params
        "task": "train",
        "objective": "regression",
        "boosting_type": "gbdt",
        "num_iterations": 500,
        "learning_rate": 0.05,
        # "learning_rate":
        #     lambda iter_n: 0.1 * (0.99 ** iter_n),
        "num_leaves": int(np.floor((2**6)*0.6)),
        "num_threads": 8,
        "seed": 42,
        # Learning Control Params
        "force_row_wise": False,
        "force_col_wise": False,
        "max_depth": 6,
        "min_data_in_leaf": 20,
        "min_sum_hessian_in_leaf": 1e-3,
        "bagging_fraction": 1.0,
        "bagging_freq": 6,
        "feature_fraction": 1.0,
        "extra_trees": False,
        "early_stopping_rounds": 200,
        "lambda_l1": 0.0,
        "malbda_l2": 0.0,
        "min_gain_to_split": 0.0,
        "path_smooth": 4,
        "verbose": -1,
        # Metric Params
        "metric": ["l1", "l2"],
        "importance_type": "gain",
    }

    data_param = {
        "max_bin": 255,
        "min_data_in_bin": 3,
    }

    train_param = {
        # "feature_name": feats_col,
        # "categorical_feature": cat_feat_cols,
        # "keep_training_booster": True,
    }

    input_len = 7
    predict_len = 1

    wlgbm = wLGBM(data_set=data,
        labels=[label_col],
        cat_feat_names=cat_feat_cols,
        is_ts=True,
        period_in=input_len,
        period_out=predict_len,
        model_name=f"wLGBM_bike_sharing_{input_len}_{predict_len}",
        feat_prefix="lbl_feat",
        multiout_mode="standalone",
        use_multiout=True,
        model_param=model_param,
        data_param=data_param,
        train_param=train_param
        )

    wlgbm.data_split(test_rate=0.2, valid_rate=0.2)

    tasks = {"categorical": cat_feat_cols,
            "one_hot_time": ["hour", "weekday", "month"],
            "sin7": ["weekday"],
            "cos7": ["weekday"],
            "sin24": ["hour"],
            "cos24": ["hour"]
        }
    processors = {"categorical": ordinal_encoder,
                "one_hot_time": one_hot_encoder,
                "sin7": sin_transformer,
                "cos7": cos_transformer,
                "sin24": sin_transformer,
                "cos24": cos_transformer,
            }
    processors_kwargs = {"sin7": {"period": 7},
            "cos7": {"period": 7},
            "sin24": {"period": 24},
            "cos24": {"period": 24},
            }
    wlgbm.preprocess_data(tasks, processors, kw_args=processors_kwargs)
    wlgbm.prepare_wide_feats()
    wlgbm.build_core_model()
    wlgbm.build_model()
    wlgbm.train_model()
    scores = wlgbm.metrics(wlgbm.predict(), is_core_model=False)
    print(f"{wlgbm.model_name} performance:")
    for name, value in socres.items():
        print(f"{name}: {value}")
