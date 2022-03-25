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

# from pathlib import Path

import numpy as np
# import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
# from sklearn.preprocessing import MinMaxScaler
from feat_transform import sin_transformer, cos_transformer
# from lightgbm import reset_parameter
import optuna

from wLGBM import wLGBM

if __name__ == "__main__":
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

    model_base_param = {
        # Core Params
        "task": "train",
        "objective": "regression",
        "num_iterations": 500,
        "num_threads": 8,
        "seed": 42,
        # Learning Control Params
        "force_row_wise": False,
        "force_col_wise": False,
        "min_sum_hessian_in_leaf": 1e-3,
        "early_stopping_rounds": 200,
        "min_gain_to_split": 0.0,
        "verbose": -1,
        # Metric Params
        "metric": ["l1", "l2"],
        # sklearn api
        "importance_type": "gain",
        }
    model_trial_param = {
        # param: ("type", Sequence[str] | [low, high])
        # Core Params
        "boosting_type":  ("categorical", ["gbdt", "dart"]),
        "learning_rate": ("float", [0.005, 10]),
        # "num_leaves": np.floor((2**6)*0.6),
        # Learning Control Params
        "max_depth": ("int", [4, 8]),
        "min_data_in_leaf": ("int", [16, 1024]),
        "bagging_fraction": ("float", [0.1, 1.0]),
        "bagging_freq": ("int", [6, 10]),
        "feature_fraction": ("float", [0.1, 1.0]),
        "extra_trees": ("categorical", [True, False]),
        "lambda_l1": ("float", [0.0, 100.0]),
        "lambda_l2": ("float", [0.0, 100.0]),
        "path_smooth": ("float", [0.0, 10.0]),
        # Metric Params
    }
    model_depend_param = {
        # param: ("type", "depend param", [low: callable, high: callable])
        # Core Params
        "num_leaves": ("float", "max_depth",
            [lambda x: 2**(x - 1), lambda x: np.floor(2**x*0.75)]),
    }
    model_condition_param = {
        # param: ("type", "condition param", "condition_value", [low, high])
        "drop_rate": ("float", "boosting_type", "dart",
                        [0.0, 1.0]),
        "skip_drop": ("float", "boosting_type", "dart",
                        [0.0, 1.0])
    }

    data_base_param = {
        "min_data_in_bin": 3,
    }
    data_trial_param = {
        "max_bin": ("int", [255, 512]),
    }

    train_base_param = {
        # "feature_name": feats_col,
        # "categorical_feature": cat_feat_cols,
        # "keep_training_booster": True, # use `init_model` in sklearn api instead,
        # "callbacks": [reset_parameter(
        #     learning_rate=lambda iter_n: model_param.get(
        #             "learning_rate", 0.1) * (0.99 ** iter_n)
        # )],
    }

    task_base_param = {
        "predict_in": 7,
        "predict_out": 3
    }
    task_trial_param = {
        # "predict_in": ("int", [7, 30]),
        # "multiout_mode": ("categorical", ["standalone", "chain"])
    }
    # input_len = 7
    # predict_len = 3

    wlgbm = wLGBM(data_set=data,
        labels=[label_col],
        cat_feat_names=cat_feat_cols,
        is_ts=True,
        # period_in=input_len,
        # period_out=predict_len,
        model_name=f"wLGBM_bike_sharing_optuna",
        feat_prefix="lbl_feat",
        multiout_mode="standalone",
        # model_param=model_param,
        # data_param=data_param,
        # train_param=train_param,
        print_info=True
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
    model_params_sets = (model_base_param,
                        model_trial_param,
                        model_depend_param,
                        model_condition_param)
    data_params_sets = (data_base_param,
                        data_trial_param,)
    # train_params_sets = (train_base_param,)
    train_params_sets = ()
    task_params_sets = (task_base_param,
                        task_trial_param)
    _ = wlgbm.initial_optimizer(
            model_params_sets=model_params_sets,
            data_params_sets=data_params_sets,
            train_params_sets=train_params_sets,
            task_params_sets=task_params_sets,
            only_train_core=True,
            metric="mean_squared_error")
    study = optuna.create_study(
                study_name="wLGBM_bike_sharing",
                direction="minimize"
            )
    study.optimize(wlgbm, n_trials=2, timeout=10*60*100)

    print(f"Numbers of finished trials: {len(study.trials)}")
    trial = study.best_trial
    print(f"Best trial:\nValue: {trial.value}")
    print("Params:")
    for k, v in trial.params.items():
        print(f"{k}: {v}")

    # save result into a file.
    with open(f"{wlgbm.model_name}.trials", "w") as file:
        file.write(f"Numbers of finished trials: {len(study.trials)}\n")
        file.write(f"Best trial:\nValue: {trial.value}\n")
        file.write("Params:\n")
        for k, v in trial.params.items():
            file.write(f"{k}: {v}\n")
