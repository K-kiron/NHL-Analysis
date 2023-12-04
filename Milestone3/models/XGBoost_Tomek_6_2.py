#!/usr/bin/env python
# coding: utf-8
"""
    This code does undersamples data with Tomek links for XGBoost model with all the data obtained from feature engineering 2.
    The metrics are then logged and the 4 required figures (ROC, Goal Rate, Cumulative Goals and Calibration display) are generated.
"""

import sys

DATA_PATH = '../../IFT6758_Data/'
PROJECT_PATH = '../../Milestone2/'

sys.path.append(PROJECT_PATH)

from comet_ml import Experiment
import numpy as np
import pandas as pd
import pickle
import imblearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from visualizations.simple_visualization import *
from models.generate_plots import *
from features.feature_eng1 import *

X = pd.read_csv(DATA_PATH + '/clean_train_data.csv', index_col=0)

has_nan = X.isna().any().any()

if has_nan:
    print("There are NaN values in the DataFrame 'X'.")
    # Dropping NaNs since these events do not have x and y coordinates
    X.dropna(inplace=True)
    X = X.reset_index(drop=True)
else:
    print("There are no NaN values in the DataFrame 'X'.")

X = X[~X.isin([np.nan, np.inf, -np.inf]).any(1)]
X = X.reset_index(drop=True)

X, y = X.iloc[:, :-1], X.iloc[:, -1]
y.value_counts(normalize=True).round(2)*100

num_cols = X.select_dtypes([np.number]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

categorical_cols = X.select_dtypes(exclude=["number", "bool_"]).columns
X = pd.get_dummies(data=X, columns=categorical_cols)

boolean_cols = X.select_dtypes([bool]).columns
X[boolean_cols] = X[boolean_cols].astype(int)
X = X.reset_index(drop=True)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

tomek = imblearn.under_sampling.TomekLinks()
X_tomek, y_tomek = tomek.fit_resample(X_train, y_train)

params = {'objective':'binary:logistic',
 'booster': 'gbtree',
 'lambda': 0.33121383940964394,
 'alpha': 7.269895266157583e-05,
 'max_depth': 9,
 'eta': 0.0965393557819441,
 'gamma': 3.264214468562427e-06,
 'grow_policy': 'depthwise'}

model = xgb.XGBClassifier(**params)

model.fit(X_tomek, y_tomek)

y_pred = model.predict(X_val)
accuracy = metrics.accuracy_score(y_val, y_pred)
print(f'Accuracy score is {accuracy}')

tomek_f1 = metrics.f1_score(y_val, y_pred)
tomek_accuracy = metrics.accuracy_score(y_val, y_pred)
tomek_precision = metrics.precision_score(y_val, y_pred)
tomek_recall = metrics.recall_score(y_val, y_pred)

pickle.dump(model, open("tomek_xgb_model.pkl", "wb"))

experiment = Experiment(
  api_key='<API_KEY>',
  project_name="nhl-project-b10",
  workspace="ift6758b-project-b10"
)

evaluation = {"accuracy": tomek_accuracy, "f1": tomek_f1, "recall": tomek_recall, "precision": tomek_precision}
params = {
    "model": 'XGBoost with Tomek Links',
    "description": 'XGBoost Classifier with Tomek Links',
    **model.get_params()
}

experiment.set_name('XGBoost with Tomek Links')
experiment.log_parameters(params)
experiment.log_metrics(evaluation)

experiment.log_model('6-2 XGBoost Classifier with Tomek Links', 'tomek_xgb_model.pkl')

experiment.end()

y_pred_prob = model.predict_proba(X_val)[:, 1]
df_percentile =  calc_percentile(y_pred_prob, y_val)

goal_rate_df = goal_rate(df_percentile)

plot_ROC(y_val, y_pred_prob, 'ROC curve for XGBoost with Tomek Links', '6-2a ROC Curve')
plot_goal_rates(goal_rate_df, 'XGBoost with Tomek Links', '6-2b Goal Rate')
plot_cumulative_goal_rates(df_percentile, 'XGBoost with Tomek Links', '6-2c Cumulative Goal Percentage')
plot_calibration_curve_prediction(y_val, y_pred_prob, '6-2d Calibration Plot')

