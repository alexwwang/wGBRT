# A window based GBRT Regressor, attempting to model time series and make prediction.

## Introduction
Use GBRT to make time series prediction, now using lightGBM as the base.
Support optuna to look for the optimize hyperparameters.

## Requirement and Install
* Python 3.8.8
* Linux/MacOS x64

Install the environment and dependencies.
```
pip install -r requirements.txt
```


## Usage
The usage of class and configuration is demonstrated in the sample file.

```
python wLGBM_optuna_sample.py
```

## Todo list
* Support other GBRT models, e.g. XGBoost, Catboost.


## Copyright
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

