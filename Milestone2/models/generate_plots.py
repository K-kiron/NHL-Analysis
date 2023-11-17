import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.calibration import  CalibrationDisplay
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import xgboost as xgb

# Reference for plot code: https://github.com/AxelBogos/NHL-Analytics/blob/master/ift6758/ift6758/models/create_figure.py
def plot_ROC(y_val, y_pred_prob, title, filenm):
    """
    Plots an ROC curve for the given y (ground truth) and model probabilities, and calculates the AUC.
    """

    fpr, tpr, _ = metrics.roc_curve(y_val, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )

    # Include a random classifier baseline, i.e. each shot has a 50% chance of being a goal
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.title(f"{title}")
    plt.legend(loc="lower right")

    ax = plt.gca()
    ax.grid()
    ax.set_facecolor('0.95')
    plt.tight_layout()
    plt.savefig(f'{filenm}.png')
    plt.show()
    plt.clf()

def goal_rate(df_percentile):

    rate_list = []

    # Find total number of goals
    total_goals = df_percentile['is_goal'].value_counts()[1]


    bin_width = 5

    i = 0
    i_list = []


    while i< (100-bin_width+1):  # 95 is the lower bound of last bin
        i_list.append(i)

        # i-th bin size
        bin_lower_bound = i
        bin_upper_bound = i + bin_width

        # finding rows have percentiles fall in this range
        bin_rows = df_percentile[(df_percentile['Percentile']>=bin_lower_bound) & (df_percentile['Percentile']<bin_upper_bound)]

        # Calculating the goal rate from total number of goals and shots in each bin_rows
        goals = bin_rows['is_goal'].value_counts()[1]
        shots = len(bin_rows) #total shots in bin_rows
        rate = (goals/shots)*100 # goal rate in pecerntage

        rate_list.append(rate)

        i+=bin_width

    # Creating a new dataframe Combining goal rate list and percentile list
    goal_rate_df = pd.DataFrame(list(zip(rate_list, i_list)),columns=['Rate', 'Percentile'])

    return goal_rate_df

def plot_goal_rates(goal_rate_df, legend, filenm):
    ax = plt.gca()
    ax.grid()

    ax.set_facecolor('0.95')
    x = goal_rate_df['Percentile']
    y = goal_rate_df['Rate']
    plt.plot(x,y)

    ax.set_ylim([0,100])
    ax.set_xlim([0,100])
    ax.invert_xaxis()
    major_ticks = np.arange(0, 110, 10)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)

    ax.legend([legend])

    plt.xlabel('Shot probability model percentile', fontsize=16)
    plt.title('Goal Rate')
    plt.ylabel('Goals / (Shots+Goals)%', fontsize=16)
    plt.savefig(f'{filenm}.png')
    plt.show()

def plot_cumulative_goal_rates(df_percentile, legend, filenm):
    df_precentile_only_goal = df_percentile[df_percentile['is_goal'] == 1]

    ax = sns.ecdfplot(data=df_precentile_only_goal, x=100 - df_precentile_only_goal.Percentile)

    ax = plt.gca()
    ax.grid()
    ax.set_facecolor('0.95')
    plt.yticks(np.arange(0, 1.05, 0.1))
    plt.xticks(np.arange(0, 100 * 1.01, 10))
    xvals = ax.get_xticks()
    ax.set_xticklabels(100 - xvals.astype(np.int32), fontsize=12)
    yvals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(y) for y in yvals], fontsize=12)
    ax.set_xlabel('Shot probability model percentile', fontsize=16)
    ax.set_ylabel('Proportion', fontsize=16)
    ax.set_title(f"Cumulative % of Goals")
    #plt.legend(loc='lower right')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.legend([legend])

    plt.savefig(f'{filenm}.png')

    plt.show()

def plot_calibration_curve_prediction(y_val, pred_probs, filenm):

    ax = CalibrationDisplay.from_predictions(y_val,pred_probs, n_bins=50)

    ax = plt.gca()
    ax.grid()
    ax.set_facecolor('0.95')
    plt.ylabel('Fraction of positives', fontsize=16)
    plt.xlabel('Mean predicted probability', fontsize=16)
    plt.savefig(f'{filenm}.png')
    plt.show()

def calc_percentile(pred_probs, y_val):

    #Create a df for shot probabilities
    df_probs = pd.DataFrame(pred_probs)
    df_probs = df_probs.rename(columns={0: 'Goal_prob'})

    # Combining 'Goal Probability' and 'Is Goal' into one df.
    df_probs = pd.concat([df_probs["Goal_prob"].reset_index(drop=True), y_val.reset_index(drop=True)],axis=1)

    # Computing and adding Percentile Column
    percentile_values=df_probs['Goal_prob'].rank(pct=True)
    df_probs['Percentile'] = percentile_values*100
    df_percentile = df_probs.copy()

    return df_percentile

def plot_roc_all_feat(y_true, y_preds, model_list, filenm, color_list, baseline = True):
    fig = plt.figure(figsize=(12, 10))

    for i, pred in enumerate(y_preds):

        plot_color = color_list[i]
        plot_label = model_list[i]

        fpr, tpr, _ = metrics.roc_curve(y_true, pred)
        roc_auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, color=plot_color, label=f'{plot_label} ' + 'AUC = %0.2f' % roc_auc, lw=2)

    # Random Baseline
    if baseline:
        baseline_is_goal = np.random.uniform(0, 1, y_preds[0].shape[0])
        plot_color = 'Magenta'
        plot_label = 'Random Baseline'
        fpr, tpr, _ = metrics.roc_curve(y_true, baseline_is_goal)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color=plot_color, label=f'{plot_label} ' + 'AUC = %0.2f' % roc_auc, lw=2)

    ax = plt.gca()
    ax.grid()
    ax.set_facecolor('0.95')

    plt.title('ROC Curves', fontsize=20)
    plt.legend(loc=2, prop={'size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{filenm}.png')
    plt.show()

def plot_goal_rate_all_feat(y_true, y_preds, model_list, filenm, color_list, baseline = True):

    fig = plt.figure(figsize=(12,10))

    for i, pred in enumerate(y_preds):

        plot_color = color_list[i]
        plot_label = model_list[i]

        df_percentile =  calc_percentile(pred, y_true)
        goal_rate_df = goal_rate(df_percentile)
        goal_rate_x = goal_rate_df['Percentile']
        goal_rate_y = goal_rate_df['Rate']
        plt.plot(goal_rate_x,goal_rate_y, color = plot_color, label = f'{plot_label}' )


    #Random Baseline
    if baseline:
        baseline_is_goal = np.random.uniform(0,1,y_preds[0].shape[0])
        no_baseline_goal = np.array([(1-i) for i in baseline_is_goal])
        random_probs = np.column_stack((baseline_is_goal, no_baseline_goal))
        df_percentile =  calc_percentile(random_probs, y_true)
        goal_rate_df = goal_rate(df_percentile)
        goal_rate_x = goal_rate_df['Percentile']
        goal_rate_y = goal_rate_df['Rate']

        plot_color = 'Magenta'
        plot_label = 'Random Baseline'
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
    plt.savefig(f'{filenm}.png')
    plt.show()

def plot_cumulative_rate_all_feat(y_true, y_preds, model_list, filenm, color_list, baseline = True):

    fig = plt.figure(figsize=(12,10))

    for i, pred in enumerate(y_preds):

        df_percentile = calc_percentile(pred, y_true)

        df_precentile_only_goal = df_percentile[df_percentile['is_goal'] == 1]

        ax = sns.ecdfplot(data=df_precentile_only_goal, x=100 - df_precentile_only_goal.Percentile,
                          color=color_list[i])

    #Random Baseline
    if baseline:
        baseline_is_goal = np.random.uniform(0, 1, y_preds[0].shape[0])
        no_baseline_goal = np.array([(1 - i) for i in baseline_is_goal])
        random_probs = np.column_stack((baseline_is_goal, no_baseline_goal))
        df_percentile = calc_percentile(random_probs, y_true)
        df_precentile_only_goal = df_percentile[df_percentile['is_goal'] == 1]

        plot_color = 'Magenta'
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

    model_list.append('Random Baseline')
    plt.legend(labels=model_list, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{filenm}.png')
    plt.show()

def plot_calibration_all_feat(y_true, y_preds, model_list, filenm, color_list):

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 3)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])

    for i, pred in enumerate(y_preds):

        plot_color = color_list[i]
        plot_label = model_list[i]

        ax_display = CalibrationDisplay.from_predictions(y_true, pred, n_bins=50,
                                                             ax=ax_calibration_curve, color=plot_color, label=plot_label)


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
    plt.savefig(f'{filenm}.png')
    plt.show()