#!/usr/bin/env python
# coding: utf-8
"""
    This code downloads all the models (3 Logistic Regression models, 1 Best XGBoost model and best model in part 6) and tests them on
    2020 Playoffs data. The metrics are then logged and the 4 required figures (ROC, Goal Rate, Cumulative Goals and Calibration display)
    are generated.
"""

import sys

DATA_PATH = '../../IFT6758_Data/'
PROJECT_PATH = '../../Milestone3/'

sys.path.append(PROJECT_PATH)

from comet_ml import API
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

from visualizations.simple_visualization import *
from models.generate_plots import *
from features.feature_eng1 import *

data = pd.read_csv(DATA_PATH + '/clean_test_data_playoffs.csv', index_col=0)

has_nan = data.isna().any().any()

if has_nan:
    print("There are NaN values in the DataFrame 'X'.")
    # Dropping NaNs since these events do not have x and y coordinates
    data.dropna(inplace=True)
    data = data.reset_index(drop=True)
else:
    print("There are no NaN values in the DataFrame 'X'.")

data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
data = data.reset_index(drop=True)

api = API(api_key='lqXuPyhuPClaazZKtvHZe5jFS')

# Logistic Regression models
api.download_registry_model('ift6758b-project-b10', 'log_reg_basemodel_angle', version='1.2.0')
api.download_registry_model('ift6758b-project-b10', 'log_reg_basemodel_distance', version='1.1.0')
api.download_registry_model('ift6758b-project-b10', 'log_reg_basemodel_distance_angle', version='1.3.0')

# XGBoost model
api.download_registry_model('ift6758b-project-b10', '5-2-hyperparameter-tuned-xgboost', version='1.1.0')

# Adaboost model
api.download_registry_model('ift6758b-project-b10', 'adaboost-max-depth-10-v3', version='1.0.3')

log_reg_distance_model = pickle.load(open('log_reg_basemodel_distance_2023-11-16 00:42:39.348668.pkl','rb'))
log_reg_angle_model = pickle.load(open('log_reg_basemodel_angle_2023-11-16 00:42:39.348668.pkl','rb'))
log_reg_distance_angle_model = pickle.load(open('log_reg_basemodel_distance_angle_2023-11-16 00:42:39.348668.pkl','rb'))

y_preds = []
distance_preds = log_reg_distance_model.predict_proba(data[['shotDistance']])[:, 1]
y_preds.append(distance_preds)

angle_preds = log_reg_angle_model.predict_proba(data[['shotAngle']])[:, 1]
y_preds.append(angle_preds)

distance_angle_preds = log_reg_distance_angle_model.predict_proba(data[['shotDistance','shotAngle']])[:, 1]
y_preds.append(distance_angle_preds)

X, y = data.iloc[:, :-1], data.iloc[:, -1]

num_cols = X.select_dtypes([np.number]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

categorical_cols = X.select_dtypes(exclude=["number", "bool_"]).columns
X = pd.get_dummies(data=X, columns=categorical_cols)

boolean_cols = X.select_dtypes([bool]).columns
X[boolean_cols] = X[boolean_cols].astype(int)
X = X.reset_index(drop=True)

xgb_hp_tune = pickle.load(open('hptuned_xgb_model.pkl','rb'))
hp_tune_preds = xgb_hp_tune.predict_proba(X)[:, 1]
y_preds.append(hp_tune_preds)

data = pd.read_csv(DATA_PATH + '/clean_test_data_playoffs_with_optional.csv', index_col=0)

has_nan = data.isna().any().any()

if has_nan:
    print("There are NaN values in the DataFrame 'X'.")
    # Dropping NaNs since these events do not have x and y coordinates
    data.dropna(inplace=True)
    data = data.reset_index(drop=True)
else:
    print("There are no NaN values in the DataFrame 'X'.")

data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
data = data.reset_index(drop=True)

X, y = data.iloc[:, :-1], data.iloc[:, -1]

num_cols = X.select_dtypes([np.number]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

categorical_cols = X.select_dtypes(exclude=["number", "bool_"]).columns
X = pd.get_dummies(data=X, columns=categorical_cols)

boolean_cols = X.select_dtypes([bool]).columns
X[boolean_cols] = X[boolean_cols].astype(int)
X = X.reset_index(drop=True)

X = X.drop(['period', 'Rebound', 'time_since_pp', 'no_players_home',
       'no_players_away', 'home_pts', 'away_pts', 'shotType_Backhand',
       'shotType_Deflected', 'shotType_Slap Shot', 'shotType_Snap Shot',
       'shotType_Tip-In', 'shotType_Wrap-around', 'shotType_Wrist Shot',
       'LastEventType_Goal', 'LastEventType_Shot'], axis=1)
adaboost_depth_10_model = pickle.load(open('ADABoost_rf_max_depth_10.pkl','rb'))
adaboost_preds = adaboost_depth_10_model.predict_proba(X)[:, 1]
y_preds.append(adaboost_preds)

color_list = ['red', 'blue', 'green', 'orange', 'magenta']
model_list = ['LogReg Distance', 'LogReg Angle', 'LogReg Distance + Angle', 'HP Tuned XGBoost', 'AdaBoost']
plot_roc_all_feat(data[['is_goal']], y_preds, model_list, '7-2a-ROC', color_list, baseline=False)
plot_goal_rate_all_feat(data[['is_goal']], y_preds, model_list, '7-2b-goal-rate', color_list, baseline=False)
plot_cumulative_rate_all_feat(data[['is_goal']], y_preds, model_list, '7-2c-cumulative', color_list, baseline=False)
plot_calibration_all_feat(data[['is_goal']], y_preds, model_list, '7-2d-calibration', color_list)

