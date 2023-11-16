import numpy as np
import pandas as pd
import xgboost as xgb
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
from sklearn.preprocessing import StandardScaler
# Read in data and assign X and y
X = pd.read_csv('../../IFT6758_Data/clean_test_data_playoffs.csv', index_col=0)
#X = data[data.columns.tolist()[:-1]]
#y_test = data[['is_goal']]
X.columns = X.columns.to_series().apply(lambda x: x.strip())

has_nan = X.isna().any().any()

if has_nan:
    print("There are NaN values in the DataFrame 'X'.")
    X.dropna(inplace=True)
    X = X.reset_index(drop=True)
else:
    print("There are no NaN values in the DataFrame 'X'.")
#X.shape
#print(X.columns)
#X.dtypes
X = X[~X.isin([np.nan, np.inf, -np.inf]).any(1)]
X = X.reset_index(drop=True)
#X = X.replace(np.inf, 0)
X, y_test = X.iloc[:, :-1], X.iloc[:, -1]
#y_test = X[['is_goal']]

#print(X['no_players_away'])
#input()
#print(X.columns)
#input()
#X = X[X.columns.tolist()[:-1]]
X = X[X.columns.tolist()]
num_cols = X.select_dtypes([np.number]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
#print(X.columns)
#print(X.columns)
#print(X['no_players_away'])
#input()
#print(X[['shotType', 'LastEventType', 'no_players_away']])
#input()
categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns
#print(categorical_cols)
#input("categorical")
X = pd.get_dummies(data=X, columns=categorical_cols)

#print(X.columns)
#input()
boolean_cols = X.select_dtypes([bool]).columns
X[boolean_cols] = X[boolean_cols].astype(int)
#print(X[boolean_cols])
#print(boolean_cols)
#input("boolean")
X = X.dropna().reset_index(drop=True)
#print(X[['shotType', 'LastEventType', 'no_players_away']])
#input()
#print(X.columns)
#input()
def pred_model(X, y_test, model, folder=""):
    if model == 'LR_D':
        best_model = joblib.load(folder + "log_reg_basemodel_distance.pkl")
        #X_test = X_test.rename({'shotDistance': 'distanceFromNet', 'shotAngle': 'angleFromNet'}, axis=1)
        X = X[['shotDistance']]
        
    elif model == 'LR_A':
        best_model = joblib.load(folder + "log_reg_basemodel_angle.pkl")
        #X_test = X_test.rename({'shotDistance': 'distanceFromNet', 'shotAngle': 'angleFromNet'}, axis=1)
        X = X[['shotAngle']]
        
    elif model == 'LR_DA':
        best_model = joblib.load(folder + "log_reg_basemodel_distance_angle.pkl")
        #X_test = X_test.rename({'shotDistance': 'distanceFromNet', 'shotAngle': 'angleFromNet'}, axis=1)
        X = X[['shotDistance', 'shotAngle']]
        
    elif model == 'HP-XGB':
        best_model = joblib.load(folder + "5-2-hptuned_xgb_model.pkl")
#        X = X [['gameSeconds', 'period', 'x_coordinate','y_coordinate', 'shotDistance',
#       'shotAngle', 'shotType', 'LastEventType', 'Last_x_coordinate',
#       'Last_y_coordinate', 'timeFromLastEvent', 'DistanceLastEvent',
#       'Rebound', 'changeShotAngle', 'speed', 'time_since_pp',
#       'no_players_home', 'no_players_away']]       
    elif model == 'feat-XGB':
        best_model = joblib.load(folder + "5-3-feat_select_xgb_model.pkl")
#        X = X [['gameSeconds', 'period', 'x_coordinate', 'y_coordinate', 'shotDistance',
#       'shotAngle', 'Last_x_coordinate','shotType', 'LastEventType', 'no_players_away'
#       'Last_y_coordinate', 'timeFromLastEvent', 'DistanceLastEvent',
#       'Rebound', 'changeShotAngle', 'speed', 'time_since_pp',
#       'no_players_home']]
        X = X.drop(['Last_x_coordinate','Last_y_coordinate','timeFromLastEvent','changeShotAngle','speed','shotType_Backhand','shotType_Deflected','shotType_Slap Shot','shotType_Snap Shot','shotType_Tip-In','shotType_Wrap-around','LastEventType_Goal','gameSeconds'], axis=1)
    elif model == 'SMOTek-XGB':
        best_model = joblib.load(folder + "6-1-SMOTek_xgb_model.pkl")
#        X = X [['gameSeconds', 'period', 'x_coordinate', 'y_coordinate', 'shotDistance',
#       'shotAngle', 'shotType', 'LastEventType', 'Last_x_coordinate',
#       'Last_y_coordinate', 'timeFromLastEvent', 'DistanceLastEvent',
#       'Rebound', 'changeShotAngle', 'speed', 'time_since_pp',
#       'no_players_home', 'no_players_away']]
    elif model == 'tomek-XGB':
        best_model = joblib.load(folder + "6-2-tomek_xgb_model.pkl")
#        X = X [['gameSeconds', 'period', 'x_coordinate', 'y_coordinate', 'shotDistance',
#       'shotAngle', 'shotType','LastEventType', 'Last_x_coordinate',
#       'Last_y_coordinate', 'timeFromLastEvent', 'DistanceLastEvent',
#       'Rebound', 'changeShotAngle','speed', 'time_since_pp',
#       'no_players_home', 'no_players_away']]
    elif model == 'smote-XGB':
        best_model = joblib.load(folder + "6-3-smote_xgb_model.pkl")
#        X = X [['gameSeconds', 'period', 'x_coordinate', 'y_coordinate', 'shotDistance',
#       'shotAngle', 'shotType', 'LastEventType', 'Last_x_coordinate',
#       'Last_y_coordinate', 'timeFromLastEvent', 'DistanceLastEvent',
#       'Rebound', 'changeShotAngle', 'speed', 'time_since_pp',
#       'no_players_home', 'no_players_away']]
    else:
        pass 

    y_pred = best_model.predict(X)
    
    #Probability estimates
    pred_probs = best_model.predict_proba(X)
    probs_isgoal = pred_probs[:,1]
    
    #Model Evaultion Metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    cf_matrix = metrics.confusion_matrix(y_test,y_pred)
    roc_auc = metrics.roc_auc_score(y_test,probs_isgoal)
    
    print(f' accuracy: {accuracy}')
    print(f' f1_score: {f1_score}')
    print(f' precision: {precision}')
    print(f' recall: {recall}')
    print(f' roc_auc: {roc_auc}')
    print('Confusion Matrix')
    print(cf_matrix)
                                
    return y_test, y_pred, accuracy, pred_probs

def plot_roc_all_feat(X, y_test):

    fig = plt.figure(figsize=(12,10))
    
    model_list = ['LR_D', 'LR_A', 'LR_DA', 'HP-XGB', 'feat-XGB', 'SMOTek-XGB', 'tomek-XGB', 'smote-XGB'] 
    model_color_list = ['red', 'blue', 'green', 'magenta', 'brown', 'cyan', "orange", "yellow"]
    #plot_label_list = ['LR_Distance', 'LR_AngleAngle from Net', 'Distance and Angle from Net', ]
    
    for i, model in enumerate(model_list):
        print(model)
        
        y_test, y_pred, accuracy,  pred_probs = pred_model(X, y_test, model, "/IFT6758B-Project-B10/IFT6758_Data/pkl-file/")
        probs_isgoal = pred_probs[:,1]
        fpr, tpr, _ = roc_curve(y_test,probs_isgoal)
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
    plt.savefig(f'7-2-2a_ROC_curves.png')
    plt.show()
plot_roc_all_feat(X, y_test)

def plot_goal_rate_all_feat(X, y_test):  
    fig = plt.figure(figsize=(12,10))
    model_list = ['LR_D', 'LR_A', 'LR_DA', 'HP-XGB', 'feat-XGB', 'SMOTek-XGB', 'tomek-XGB', 'smote-XGB'] 
    model_color_list = ['red', 'blue', 'green', 'magenta', 'brown', 'cyan', "orange", "yellow"]
    
    for i, model in enumerate(model_list):
        print(model)
        
        y_test, y_pred, accuracy,  pred_probs = pred_model(X, y_test, model, "/IFT6758B-Project-B10/IFT6758_Data/pkl-file/")  
        df_percentile =  calc_percentile(pred_probs, y_test)

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
    plt.savefig(f'7-2-2b_goal_rates.png')
    plt.show()
plot_goal_rate_all_feat(X, y_test)
def plot_cumulative_rate_all_feat(X_test, y_test):

    fig = plt.figure(figsize=(12,10))
    model_list = ['LR_D', 'LR_A', 'LR_DA', 'HP-XGB', 'feat-XGB', 'SMOTek-XGB', 'tomek-XGB', 'smote-XGB'] 
    model_color_list = ['red', 'blue', 'green', 'magenta', 'brown', 'cyan', "orange", "yellow"]
    
    for i, model in enumerate(model_list):
        print(model)

        
        y_test, y_pred, accuracy,  pred_probs = pred_model(X, y_test, model, "/IFT6758B-Project-B10/IFT6758_Data/pkl-file/") 
        df_percentile = calc_percentile(pred_probs, y_test)    
        df_precentile_only_goal = df_percentile[df_percentile['is_goal'] == 1]
        
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
    plt.savefig(f'7-2-2c_goal_proportions.png')
    plt.show()
plot_cumulative_rate_all_feat(X, y_test)
def plot_calibration_all_feat(X, y_test):

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 3)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    
    model_list = ['LR_D', 'LR_A', 'LR_DA', 'HP-XGB', 'feat-XGB', 'SMOTek-XGB', 'tomek-XGB', 'smote-XGB'] 
    model_color_list = ['red', 'blue', 'green', 'magenta', 'brown', 'cyan', "orange", "yellow"]
    
    for i, model in enumerate(model_list):
        print(model)

        y_test, y_pred, accuracy,  pred_probs = pred_model(X, y_test, model, "/IFT6758B-Project-B10/IFT6758_Data/pkl-file/")
        y_test_is_goal = y_test
        probs_isgoal = pred_probs[:,1]
 
        plot_color = model_color_list[i]
        plot_label = model_list[i] 
        ax_display = CalibrationDisplay.from_predictions(y_test_is_goal,probs_isgoal, n_bins=50,ax=ax_calibration_curve, color=plot_color, label=plot_label)                                         
    ax = plt.gca()
    ax_calibration_curve.grid()
    ax.set_facecolor('0.95')    

    plt.title("Calibration plots", fontsize=20)
    plt.legend(loc=2,prop={'size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.ylabel('Fraction of positives', fontsize=20)
    plt.xlabel('Mean predicted probability', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'7-2-2d_calibration_plots.png')
    plt.show()
plot_calibration_all_feat(X, y_test)


