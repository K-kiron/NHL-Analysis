#!/usr/bin/env python
# coding: utf-8
"""
    This code trains the baseline XGBoost model with all permutations of shot distance and
    shot angle features. The metrics are then logged and the 4 required figures (ROC, Goal Rate,
    Cumulative Goals and Calibration display) are generated.
"""

import sys

DATA_PATH = '../../IFT6758_Data/'
PROJECT_PATH = '../../Milestone2/'


sys.path.append(PROJECT_PATH)

from comet_ml import Experiment
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

from visualizations.simple_visualization import *
from models.generate_plots import *
from features.feature_eng1 import *

data = pd.read_csv(DATA_PATH + '/clean_train_data.csv', index_col=0)

X = data[['shotDistance', 'shotAngle', 'is_goal' ]]

has_nan = X.isna().any().any()

if has_nan:
    print("There are NaN values in the DataFrame 'X'.")
    
    # Dropping NaNs since these events do not have x and y coordinates
    X.dropna(inplace=True)
    X = X.reset_index(drop=True)
else:
    print("There are no NaN values in the DataFrame 'X'.")


X, y = X.iloc[:, :-1], X.iloc[:, -1]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = xgb.XGBClassifier(objective='binary:logistic')
model.fit(X_train, y_train)

# Compute metrics
y_pred = model.predict(X_val)
f1 = f1_score(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
params = {
    "model": 'XGBoost',
    "description": 'XGBClassifier Baseline Distance+Angle',
}

experiment = Experiment(
  api_key='API_KEY',
  project_name="nhl-project-b10",
  workspace="ift6758b-project-b10"
)

experiment.set_name('XGBoost Baseline')
experiment.log_parameters(params)
experiment.log_metrics(metrics)

experiment.end()

feature_list = (['shotDistance'], ['shotAngle'], ['shotDistance', 'shotAngle'])
model_list = ['Distance from Net', 'Angle from Net', 'Distance and Angle from Net']
color_list = ['red', 'blue', 'green']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

y_preds = []

for i, feature in enumerate(feature_list):
    
    # Fit the model
    model.fit(X_train[feature], y_train)
    
    # Obtain the predictions and prediction probabilities
    y_pred_prob = model.predict_proba(X_val[feature])
    
    y_val_is_goal = y_val
    y_preds.append(y_pred_prob[:,1])

plot_roc_all_feat(y_val, y_preds, model_list, '5-1a ROC Curves', color_list)
plot_goal_rate_all_feat(y_val, y_preds, model_list, '5-1b Goal rate', color_list)
plot_cumulative_rate_all_feat(y_val, y_preds, model_list, '5-1c Cumulative Goal Percentage', color_list)
plot_calibration_all_feat(y_val, y_preds, model_list, '5-1d Calibration Plots', color_list)

