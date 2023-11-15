import numpy as np
import pandas as pd
#import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve, CalibrationDisplay
import joblib
from plots import *
from matplotlib.gridspec import GridSpec
import pickle


# Read in data and assign X and y
#data = pd.read_csv('../../IFT6758_Data/clean_test_data_playoff.csv', index_col=0)
data = pd.read_csv('../../IFT6758_Data/clean_test_data_regular.csv', index_col=0)

print(data.head())
X_test = data[data.columns.tolist()[:-1]]
y_test = data[['is_goal']]

def pred_model(X_test, y_test, model, folder=""):
    if model == 'LR_D':
        best_model = joblib.load(folder + "log_reg_basemodel_distance.pkl")
        X_test = X_test.rename({'shot_distance': 'distanceFromNet', 'shot_angle': 'angleFromNet'}, axis=1)
        X_test = X_test[['distanceFromNet']]
        
    elif model == 'LR_A':
        best_model = joblib.load(folder + "log_reg_basemodel_angle.pkl")
        X_test = X_test.rename({'shot_distance': 'distanceFromNet', 'shot_angle': 'angleFromNet'}, axis=1)
        X_test = X_test[['angleFromNet']]
        
    elif model == 'LR_DA':
        best_model = joblib.load(folder + "log_reg_basemodel_distance_angle.pkl")
        X_test = X_test.rename({'shot_distance': 'distanceFromNet', 'shot_angle': 'angleFromNet'}, axis=1)
        X_test = X_test[['distanceFromNet', 'angleFromNet']]
        
#    elif model == 'XGB':
#        best_model = joblib.load(folder + ".pkl")
       
#    elif model == 'RF':
#        best_model = joblib.load(folder + ".pkl")
        
#    elif model == 'DT':
#        best_model = joblib.load(folder + ".pkl")
    
    # elif model == 'NN':
        # best_model = joblib.load(folder + "nn_weighted.h5")
#    
#    elif model == 'SVM':
#        best_model = joblib.load(folder + "svm_best_prob.pkl")
        
    else:
        pass 

    y_pred = best_model.predict(X_test)
    
    #Probability estimates
    pred_probs = best_model.predict_proba(X_test)
    probs_isgoal = pred_probs[:,1]
    
    #Model Evaultion Metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    #precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    cf_matrix = metrics.confusion_matrix(y_test,y_pred)
    roc_auc = metrics.roc_auc_score(y_test,probs_isgoal)
    
    #print(f' accuracy: {accuracy}')
    #print(f' f1_score: {f1_score}')
    #print(f' precision: {precision}')
    #print(f' recall: {recall}')
    #print(f' roc_auc: {roc_auc}')
    #print('Confusion Matrix')
    #print(cf_matrix)
                                
    return y_test, y_pred, accuracy, pred_probs
"""
def plot_roc_all_feat(X_test, y_test):

    fig = plt.figure(figsize=(12,10))
    
    model_list = ['LR_D', 'LR_A', 'LR_DA', 'XGB', 'RF', 'DT', "NN", "SVM"] 
    model_color_list = ['red', 'blue', 'green', 'magenta', 'brown', 'cyan', "orange", "yellow"]
    #plot_label_list = ['LR_Distance', 'LR_AngleAngle from Net', 'Distance and Angle from Net', ]
    
    for i, model in enumerate(model_list):
        print(model)
        
        if model != "NN":
            y_test, y_pred, accuracy,  pred_probs = pred_model(X_test, y_test, model, "../model/")
            probs_isgoal = pred_probs[:,1]
            fpr, tpr, _ = roc_curve(y_test,probs_isgoal)
        else:
            file = open("../6_Different_Models/NN/results_nn_regular.pkl",'rb')
            res = pickle.load(file)
            file.close()
            fpr, tpr, _ = roc_curve(y_test, res["y_"])
        roc_auc = auc(fpr,tpr)
        
        plot_color = model_color_list[i]
        plot_label = model_list[i]
        plt.plot(fpr, tpr, color = plot_color, label = f'{plot_label} '+'AUC = %0.2f' % roc_auc, lw=2)               
    
    ax = plt.gca()
    ax.grid()
    ax.set_facecolor('0.95')
    
    plt.title('ROC Curves', fontsize=20)
    plt.legend(loc=2,prop={'size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'7-1-1a_ROC_curves.png')
    plt.show()
plot_roc_all_feat(X_test, y_test)
def plot_goal_rate_all_feat(X_test, y_test):  
    fig = plt.figure(figsize=(12,10))
    model_list = ['LR_D', 'LR_A', 'LR_DA', 'XGB', 'RF', 'DT', "NN", "SVM"] 
    model_color_list = ['red', 'blue', 'green', 'magenta', 'brown', 'cyan', "orange", "yellow"]
    
    for i, model in enumerate(model_list):
        print(model)
        if model != "NN":
            y_test, y_pred, accuracy,  pred_probs = pred_model(X_test, y_test, model, "../model/")  
            df_percentile =  calc_percentile(pred_probs, y_test)
        else:
            file = open("../6_Different_Models/NN/results_nn_regular.pkl",'rb')
            res = pickle.load(file)
            file.close()
            df_percentile = calc_percentile(res["pred_probs"], y_test)

        goal_rate_df = goal_rate(df_percentile)
        goal_rate_x = goal_rate_df['Percentile']
        goal_rate_y = goal_rate_df['Rate']
        plot_color = model_color_list[i]
        plot_label = model_list[i]
        plt.plot(goal_rate_x,goal_rate_y, color = plot_color, label = f'{plot_label}' )
                 
       
    ax = plt.gca()
    ax.grid()
    ax.set_facecolor('0.95')
    
    ax.set_ylim([0,100])
    ax.set_xlim([0,100])
    ax.invert_xaxis()
    major_ticks = np.arange(0, 110, 10)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    
    plt.grid(True)
    plt.title('Goal Rate', fontsize=20)
    plt.xlabel('Shot probability model percentile', fontsize=16)
    plt.ylabel('Goals / (Shots+Goals)%', fontsize=16)
    plt.legend(loc=2,prop={'size': 16})
    plt.tight_layout()
    plt.savefig(f'7-1-1b_goal_rates.png')
    plt.show()
plot_goal_rate_all_feat(X_test, y_test)
def plot_cumulative_rate_all_feat(X_test, y_test):

    fig = plt.figure(figsize=(12,10))
    model_list = ['LR_D', 'LR_A', 'LR_DA', 'XGB', 'RF', 'DT', "NN", "SVM"] 
    model_color_list = ['red', 'blue', 'green', 'magenta', 'brown', 'cyan', "orange", "yellow"]
    
    for i, model in enumerate(model_list):
        print(model)

        if model != "NN":
            y_test, y_pred, accuracy,  pred_probs = pred_model(X_test, y_test, model, "../model/") 
            df_percentile = calc_percentile(pred_probs, y_test)
        else:
            file = open("../6_Different_Models/NN/results_nn_regular.pkl",'rb')
            res = pickle.load(file)
            file.close()
            df_percentile = calc_percentile(res["pred_probs"], y_test)
            
        df_precentile_only_goal = df_percentile[df_percentile['isGoal'] == 1]
        
        plot_color = model_color_list[i]
        plot_label = model_list[i]
        ax = sns.ecdfplot(data=df_precentile_only_goal, x=100 - df_precentile_only_goal.Percentile, 
                              color=plot_color)
            
    ax = plt.gca()
    ax.grid()
    ax.set_facecolor('0.95')
    plt.yticks(np.arange(0, 1.05, 0.1))
    plt.xticks(np.arange(0, 100 * 1.01, 10))
    xvals = ax.get_xticks()
    ax.set_xticklabels(100 - xvals.astype(np.int32), fontsize=16)
    yvals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(y) for y in yvals], fontsize=16)
    ax.set_xlabel('Shot probability model percentile', fontsize=16)
    ax.set_ylabel('Proportion', fontsize=16)
    ax.set_title(f"Cumulative % of Goals", fontsize=20)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    
    
    plt.legend(labels=model_list, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'7-1-1c_goal_proportions.png')
    plt.show()
plot_cumulative_rate_all_feat(X_test, y_test)
def plot_calibration_all_feat(X_test, y_test):

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 3)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    
    model_list = ['LR_D', 'LR_A', 'LR_DA', 'XGB', 'RF', 'DT', "NN", "SVM"] 
    model_color_list = ['red', 'blue', 'green', 'magenta', 'brown', 'cyan', "orange", "yellow"]
    
    for i, model in enumerate(model_list):
        print(model)

        if model != "NN":
            y_test, y_pred, accuracy,  pred_probs = pred_model(X_test, y_test, model, "../model/")
            y_test_is_goal = y_test['isGoal']
            probs_isgoal = pred_probs[:,1]
        else:
            file = open("../6_Different_Models/NN/results_nn_regular.pkl",'rb')
            res = pickle.load(file)
            file.close()
            probs_isgoal = res["y_"]

        plot_color = model_color_list[i]
        plot_label = model_list[i] 
        ax_display = CalibrationDisplay.from_predictions(y_test_is_goal,probs_isgoal, n_bins=50,                                         
    ax = plt.gca()
    ax.set_facecolor('0.95')
    ax_calibration_curve.grid()
    
    plt.title("Calibration plots", fontsize=20)
    plt.legend(loc=2,prop={'size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.ylabel('Fraction of positives', fontsize=20)
    plt.xlabel('Mean predicted probability', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'7-1-1d_calibration_plots.png')
    plt.show()
plot_calibration_all_feat(X_test, y_test)
"""
