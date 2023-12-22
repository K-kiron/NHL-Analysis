#!/usr/bin/env python
# coding: utf-8
"""
    This code does hyperparameter tuning of XGBoost model with all the data obtained from feature engineering2.
    The metrics are then logged and the 4 required figures (ROC, Goal Rate, Cumulative Goals and Calibration display)
    are generated.
"""

import sys

DATA_PATH = '../../IFT6758_Data/'
PROJECT_PATH = '../../Milestone3/'

sys.path.append(PROJECT_PATH)

from comet_ml import Experiment
import numpy as np
import pickle
import pandas as pd
import optuna
import sklearn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
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

num_cols = X.select_dtypes([np.number]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

categorical_cols = X.select_dtypes(exclude=["number", "bool_"]).columns
X = pd.get_dummies(data=X, columns=categorical_cols)

boolean_cols = X.select_dtypes([bool]).columns
X[boolean_cols] = X[boolean_cols].astype(int)
X = X.reset_index(drop=True)

# Reference for using optuna with XGBoost: https://medium.com/optuna/using-optuna-to-optimize-xgboost-hyperparameters-63bfcdfd3407
def objective(X, y, trial):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    param_values = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0)
    }

    if param_values["booster"] == "gbtree" or param_values["booster"] == "dart":
        param_values["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param_values["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param_values["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param_values["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param_values["booster"] == "dart":
        param_values["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param_values["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param_values["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param_values["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
    bst = xgb.train(param_values, dtrain, evals=[(dval, "validation")], callbacks=[pruning_callback])
    preds = bst.predict(dval)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_val, pred_labels)
    return accuracy

study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                                direction="maximize", study_name="XGB Classifier")

func = lambda trial: objective(X, y, trial)
study.optimize(func, n_trials=100)

# Save study plots
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig.write_html('5-2_optuna_history.html')
fig = optuna.visualization.plot_param_importances(study)
fig.show()
fig.write_html('5-2_hp_imp.html')

params = study.best_params
params = {'objective':'binary:logistic',
 'booster': 'gbtree',
 'lambda': 0.33121383940964394,
 'alpha': 7.269895266157583e-05,
 'max_depth': 9,
 'eta': 0.0965393557819441,
 'gamma': 3.264214468562427e-06,
 'grow_policy': 'depthwise'}

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_val)[:, 1]

y_pred = model.predict(X_val)
accuracy = metrics.accuracy_score(y_val, y_pred)
print(f'Accuracy score is {accuracy}')

df_percentile =  calc_percentile(y_pred_prob, y_val)

goal_rate_df = goal_rate(df_percentile)

plot_ROC(y_val, y_pred_prob, 'ROC curve for Hyperparameter tuned XGBoost', '5-2a ROC Curve')
plot_goal_rates(goal_rate_df, 'Hyperparameter tuned XGBoost', '5-2b Goal Rate')
plot_cumulative_goal_rates(df_percentile, 'Hyperparameter tuned XGBoost', '5-2c Cumulative Goal Percentage')
plot_calibration_curve_prediction(y_val, y_pred_prob, '5-2d Calibration Plot')

pickle.dump(model, open("hptuned_xgb_model.pkl", "wb"))

experiment = Experiment(
  api_key='API_KEY',
  project_name="nhl-project-b10",
  workspace="ift6758b-project-b10"
)

f1 = f1_score(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
evaluation = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
params = {
    "model": 'Tuned XGBoost',
    "description": 'Hyperparameter tuned XGBoost Classifier',
    **model.get_params()
}


experiment.set_name('XGBoost with Hyperparameter Tuning')
experiment.log_parameters(params)
experiment.log_metrics(evaluation)

experiment.log_model('5-2 Hyperparameter tuned XGBoost Classifier', 'hptuned_xgb_model.pkl')

experiment.end()

