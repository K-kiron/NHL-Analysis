from comet_ml import Experiment
import os 
import datetime
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from plots import *


now = datetime.datetime.now()

# Read in data and assign X and y
data = pd.read_csv('../../../IFT6758_Data/train_data.csv', index_col=0)
data = data[['shotDistance', 'shotAngle','is_goal' ]]
data.dropna(inplace=True)
X = data[['shotDistance', 'shotAngle' ]]
X = X.rename({'shotDistance': 'distanceFromNet', 'shotAngle': 'angleFromNet'}, axis=1)
y = data[['is_goal']]

def Log_reg(X, y, i):
    """
    Trains a logistic regression model using specified features from the input data, evaluates its performance,
    and logs the experiment results to Comet.ml. The function also generates and saves various plots such as
    ROC curve, goal rate plot, cumulative goal rate plot, and calibration curve.

    The model is trained on the features specified by the index 'i', which selects from predefined feature sets.

    Parameters:
    - X (pd.DataFrame): DataFrame containing the input features for the model.
    - y (pd.Series): Series containing the binary target variable for the model.
    - i (int): Index to select the feature set for training the model. Possible values are:
        0 for 'Distance from Net',
        1 for 'Angle from Net',
        2 for 'Distance and Angle from Net'.

    Returns:
    A tuple containing:
    - pred_probs (np.ndarray): The probability estimates for the validation set.
    - accuracy (float): The accuracy of the model on the validation set.
    - f1_score (float): The F1 score of the model on the validation set.
    - precision (float): The precision of the model on the validation set.
    - recall (float): The recall of the model on the validation set.
    - roc_auc (float): The ROC AUC score of the model on the validation set.
    - cf_matrix (np.ndarray): The confusion matrix of the model on the validation set.
    """
    feature_list = (['distanceFromNet'], ['angleFromNet'], ['distanceFromNet', 'angleFromNet']  )
    feature_name_list = ['distance', 'angle', 'distance_angle']
    
    experiment = Experiment(
        api_key="COMET_API_KEY",  
        project_name="nhl-project-b10",
        workspace="ift6758b-project-b10",
        auto_output_logging="simple",
        )
    features = feature_list[i]
    feature_name = feature_name_list[i]
        
    # set an experiment name for basemodel
    experiment_name = "log_reg_basemodel_" + feature_name +"_" + str(now) #base name for log_model, log_image
    experiment.set_name(experiment_name)
    #add tags
    experiment.add_tags([feature_name])
        

    #Train and valid split
        
    X_ = X[features].to_numpy()
    min_x = np.min(X_,axis=0)
    max_x = np.max(X_,axis=0)
    X_ = X_ - min_x[None,:] / ( max_x[None,:] - min_x[None,:]) 

    y_ = y.to_numpy().reshape(-1)
    X_train,X_val,y_train,y_val = train_test_split(X_, y_, test_size=0.2, random_state=42)

       

    # Logistic regression model fitting
    clf = LogisticRegression(class_weight='balanced', C=40)
    #y_train = y_train.values.ravel()
    clf.fit(X_train, y_train)
        
    # Predict on validation set
    y_pred = clf.predict(X_val)
        
    #Probability estimates
    pred_probs = clf.predict_proba(X_val)
        
    #Model Evaultion Metrics
    accuracy = metrics.accuracy_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    cf_matrix = metrics.confusion_matrix(y_val,y_pred)
        
    #ROC AUC Curve
    probs_isgoal = pred_probs[:,1]
    roc_auc = metrics.roc_auc_score(y_val,probs_isgoal)
    plot_ROC(y_val, pred_probs)
        
    #Goal Rate Plot
    df_percentile =  calc_percentile(pred_probs, y_val)
    goal_rate_df = goal_rate(df_percentile)
    plot_goal_rates(goal_rate_df)
        
    #Cumulative Goal Rate Plot
    plot_cumulative_goal_rates(df_percentile)
        
    #Calibration Curve
    plot_calibration_curve_prediction(y_val, pred_probs)

    # save the model to disk
    filename = experiment_name + '.pkl'
    pickle.dump(clf, open(filename, 'wb')) 
        
    #params = {}
    metrics_dict = { 'accuracy': accuracy,
                    "f1_score": f1_score,
                    "precision": precision,
                    "recall": recall,
                    "roc_auc": roc_auc}

    experiment.log_metrics(metrics_dict)
    experiment.log_confusion_matrix(matrix=cf_matrix)
    experiment.log_image('roc_curve.png', name= experiment_name + '_roc_curve.png', overwrite=True)
    experiment.log_image('goal_rate_plot.png', name= experiment_name + '_goal_rate_plot.png', overwrite=True)
    experiment.log_image('cumulative_goal_rate.png', name= experiment_name + '_cumulative_goal_rate_plot.png', overwrite=True)
    experiment.log_image('calibration_curve.png', name= experiment_name + '_calibration_curve.png', overwrite=True)
    experiment.log_model(experiment_name, filename)
           
    return pred_probs, accuracy,f1_score, precision, recall, roc_auc, cf_matrix
    

if __name__ == '__main__':
    i = 0
    while i<3 : 
        pred_probs, accuracy,f1_score, precision, recall, roc_auc, cf_matrix = Log_reg(X, y, i)
        print(accuracy,f1_score, precision, recall, roc_auc )
        print(cf_matrix)
        i+=1

    
    


