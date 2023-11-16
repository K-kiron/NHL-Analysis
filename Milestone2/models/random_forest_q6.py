#!/usr/bin/env python
# coding: utf-8

# In[1]:


from comet_ml import Experiment
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel


# In[2]:


#DATA_PATH = "/Users/tristanmartin/Desktop/UdeM_PhD/Cours/A2023/IFT6758/Project/IFT6758B-Project-B10-main-2/Milestone2/data"
#PROJECT_PATH = "/Users/tristanmartin/Desktop/UdeM_PhD/Cours/A2023/IFT6758/Project/IFT6758B-Project-B10-main-2/Milestone2/"

DATA_PATH = '../../IFT6758_Data/'
PROJECT_PATH = '../../Milestone2/'

import sys
sys.path.append(PROJECT_PATH)

from features.feature_eng2 import *
from features.tidy_data import *
from features.feature_eng1 import *
from visualizations.simple_visualization import *
from models.BaselineModels.plots import *


# In[3]:


#get_train_data(DATA_PATH)


# In[4]:


# Loading data and pre-processing
#X = pd.read_csv(DATA_PATH + '/Users/tristanmartin/Desktop/UdeM_PhD/Cours/A2023/IFT6758/Project/IFT6758B-Project-B10-main-2/Milestone2/data/clean_train_data.csv', index_col=0)
X = pd.read_csv('/Users/tristanmartin/Desktop/UdeM_PhD/Cours/A2023/IFT6758/Project/IFT6758B-Project-B10-main-2/Milestone2/data/clean_train_data.csv', index_col=0)

has_nan = X.isna().any().any()

if has_nan:
    print("There are NaN values in the DataFrame 'X'.")
    # Dropping NaNs since these events do not have x and y coordinates
    X.dropna(inplace=True)
    X = X.reset_index(drop=True)
else:
    print("There are no NaN values in the DataFrame 'X'.")

X = X[~X.isin([np.nan, np.inf, -np.inf]).any(axis = 1)]
X = X.reset_index(drop=True)

X, y = X.iloc[:, :-1], X.iloc[:, -1]

num_cols = X.select_dtypes([np.number]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns
X = pd.get_dummies(data=X, columns=categorical_cols)

boolean_cols = X.select_dtypes([bool]).columns
X[boolean_cols] = X[boolean_cols].astype(int)
X = X.reset_index(drop=True)


# In[13]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

rf = RandomForestClassifier(random_state=42)
sfm = SelectFromModel(rf)

X_res = sfm.fit_transform(X_train, y_train)
X_val_selected = sfm.transform(X_val)

params = {'n_estimators': 1,  # You can adjust the number of weak learners
          'learning_rate': 1.0,
          'algorithm': 'SAMME.R',
          'random_state': 42,
          # Try max_depth = 1,3,5
          'estimator': RandomForestClassifier(max_depth=10),
          }

model = AdaBoostClassifier(**params)

sample_weights = len(y_train) / (2 * np.bincount(y_train))
sample_weights /= sum(sample_weights)

model.fit(X_res, y_train, sample_weight=sample_weights[y_train])

y_pred = model.predict(X_val_selected)


# In[15]:


f1 = f1_score(y_val, y_pred)
print(f'f1 score: {f1}')
accuracy = accuracy_score(y_val, y_pred)
print(f'accuracy score: {accuracy}')
precision = precision_score(y_val, y_pred)
print(f'precision score: {precision}')
recall = recall_score(y_val, y_pred)
print(f'recall score: {recall}')


# In[16]:


pickle.dump(model, open("ADABoost_rf_max_depth_10.pkl", "wb"))
experiment = Experiment(
  api_key='M0ld212AYoT5RG6UcLL807o5T',
  project_name="nhl-project-b10",
  workspace="ift6758b-project-b10"
)

evaluation = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
params = {
    "model": 'ADABoost',
    "description": 'ADABoost Classifier with Max Deep = 10 and Random Forest Feature Selection on Feature Eng2 Cleaned Dataframe',
    **model.get_params()
}
experiment.set_name('ADABoost Max Depth = 10')
experiment.log_parameters(params)
experiment.log_metrics(evaluation)

experiment.log_model('ADABoost Max Depth = 10', 'ADABoost_rf_max_depth_10.pkl') #Edit this
experiment.end() # Important if you are using jupyter


# In[ ]:




