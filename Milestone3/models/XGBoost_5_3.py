#!/usr/bin/env python
# coding: utf-8
"""
    This code does feature selection using VarianceThreshold, SelectKBest and ExtraTreesClassifier for XGBoost model with all the data obtained
    from feature engineering2. The metrics are then logged and the 4 required figures (ROC, Goal Rate, Cumulative Goals and Calibration display)
    are generated.
"""

import sys

DATA_PATH = '../../IFT6758_Data/'
PROJECT_PATH = '../../Milestone3/'

sys.path.append(PROJECT_PATH)

from comet_ml import Experiment
import numpy as np
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, SelectFromModel
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

num_cols = X.select_dtypes([np.number]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

categorical_cols = X.select_dtypes(exclude=["number", "bool_"]).columns
X = pd.get_dummies(data=X, columns=categorical_cols)

boolean_cols = X.select_dtypes([bool]).columns
X[boolean_cols] = X[boolean_cols].astype(int)
X = X.reset_index(drop=True)

# Use best params from previous question
params = {'objective':'binary:logistic',
 'booster': 'gbtree',
 'lambda': 0.33121383940964394,
 'alpha': 7.269895266157583e-05,
 'max_depth': 9,
 'eta': 0.0965393557819441,
 'gamma': 3.264214468562427e-06,
 'grow_policy': 'depthwise'}

xgb_model = xgb.XGBClassifier(**params)

# Reference: https://scikit-learn.org/stable/modules/feature_selection.html

# Remove features with low variance
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
var_threshold_fold_score = []

for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_train_var_threshold = sel.fit_transform(X_train)
    xgb_model.fit(X_train_var_threshold, y_train)
    
    X_val_var_threshold = sel.transform(X_val)
    y_pred = xgb_model.predict(X_val_var_threshold)
    f1_score = metrics.f1_score(y_val, y_pred)
    
    var_threshold_fold_score.append(f1_score)

var_threshold_score = np.mean(var_threshold_fold_score)
var_threshold_score

# Select K best features
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
k_best_fold_score = []

for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    sel = SelectKBest(score_func=mutual_info_classif, k=12)
    X_train_var_threshold = sel.fit_transform(X_train, y_train)
    
    xgb_model.fit(X_train_var_threshold, y_train)
    
    X_val_var_threshold = sel.transform(X_val)
    y_pred = xgb_model.predict(X_val_var_threshold)
    f1_score = metrics.f1_score(y_val, y_pred)
    
    k_best_fold_score.append(f1_score)

k_best_score = np.mean(k_best_fold_score)
k_best_score

# Tree based feature selection
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
tree_fold_score = []

for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_train, y_train)
    sel = SelectFromModel(clf, prefit=True)
    X_train_var_threshold = sel.transform(X_train)
    
    xgb_model.fit(X_train_var_threshold, y_train)
    
    X_val_var_threshold = sel.transform(X_val)
    y_pred = xgb_model.predict(X_val_var_threshold)
    f1_score = metrics.f1_score(y_val, y_pred)
    
    tree_fold_score.append(f1_score)
    
tree_selection_score = np.mean(tree_fold_score)
tree_selection_score

# Reference: https://github.com/shap/shap

xgb_fit_model = xgb_model.fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(xgb_fit_model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])

plt.close()

shap.initjs()
shap.plots.force(shap_values[0])

# Reference: https://medium.com/nerd-for-tech/removing-constant-variables-feature-selection-463e2d6a30d9

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X_train, y_train)
sel = SelectFromModel(clf, prefit=True)
sel.transform(X_train)

concol = [column for column in X_train.columns if column not in X_train.columns[sel.get_support()]]


for features in concol:
    print('features')

X_train = X_train.drop(concol,axis=1)
X_val = X_val.drop(concol,axis=1)

params = {'objective':'binary:logistic',
 'booster': 'gbtree',
 'lambda': 0.33121383940964394,
 'alpha': 7.269895266157583e-05,
 'max_depth': 9,
 'eta': 0.0965393557819441,
 'gamma': 3.264214468562427e-06,
 'grow_policy': 'depthwise'}

xgb_model = xgb.XGBClassifier(**params)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_val)
accuracy = metrics.accuracy_score(y_val, y_pred)
print(f'Accuracy score is {accuracy}')

pickle.dump(xgb_model, open("feat_select_xgb_model.pkl", "wb"))

experiment = Experiment(
  api_key='<API_KEY>',
  project_name="nhl-project-b10",
  workspace="ift6758b-project-b10"
)

f1 = metrics.f1_score(y_val, y_pred)
accuracy = metrics.accuracy_score(y_val, y_pred)
precision = metrics.precision_score(y_val, y_pred)
recall = metrics.recall_score(y_val, y_pred)
evaluation = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision, 
              "variance_threshold_score": var_threshold_score ,"k_best_score": k_best_score, 
              "tree_based_selection_score": tree_selection_score }
params = {
    "model": 'XGBoost with Feature Selection',
    "description": 'XGBoost Classifier with Variance Thresholding Feature Selection',
    **xgb_model.get_params()
}

experiment.set_name('XGBoost with Feature Selection')
experiment.log_parameters(params)
experiment.log_metrics(evaluation)

experiment.log_model('5-3 Feature Selection XGBoost Classifier', 'feat_select_xgb_model.pkl')
experiment.end()

y_pred_prob = xgb_model.predict_proba(X_val)[:, 1]
df_percentile =  calc_percentile(y_pred_prob, y_val)

goal_rate_df = goal_rate(df_percentile)

plot_ROC(y_val, y_pred_prob, 'ROC curve for XGBoost with Feature Selection', '5-3a ROC Curve')
plot_goal_rates(goal_rate_df, 'XGBoost with Feature Selection', '5-3b Goal Rate')
plot_cumulative_goal_rates(df_percentile, 'XGBoost with Feature Selection', '5-3c Cumulative Goal Percentage')
plot_calibration_curve_prediction(y_val, y_pred_prob, '5-3d Calibration Plot')

