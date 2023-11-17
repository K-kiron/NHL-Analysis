import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.ticker as ticker



def plot_ROC(y_val,pred_probs):
    """
    Generates and displays a Receiver Operating Characteristic (ROC) curve with the corresponding Area Under 
    the Curve (AUC) metric from true binary labels and prediction probabilities. Optionally, it can save the
    plot to a file.

    The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) across different 
    thresholds. The function also plots a diagonal line representing the performance of a random classifier 
    for reference. The AUC value is calculated and displayed in the legend.

    Parameters:
    - y_val (array-like): True binary labels in range {0, 1} or {-1, 1}.
    - probs (array-like): Probability estimates of the positive class, returned by a classifier.
    - title (str, optional): Title for the plot. If not specified, no title is added.
    - savename (str, optional): If provided, the plot will be saved to a file with this name.

    Returns:
    None

    """    
    #Plots an ROC curve for the given y (ground truth) and model probabilities, and calculates the AUC.
    
    y_val = pd.DataFrame(y_val)
    y_val = y_val.rename(columns={0: "is_goal"})

    probs_isgoal = pred_probs[:,1]
    fpr, tpr, _ = roc_curve(y_val,probs_isgoal)
    roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(8,6))
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    #Include a random classifier baseline, i.e. each shot has a 50% chance of being a goal
    
    
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.title('ROC Curves', fontsize=16)
    plt.legend(loc="lower right")
    
    ax = plt.gca()
    ax.grid()
    ax.set_facecolor('0.95')
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    
def calc_percentile(pred_probs, y_val):
    """
    Calculates the percentile rank of predicted probabilities and merges it with the actual goal labels.

    This function takes the predicted probabilities for the positive class (goals) and the actual 
    labels (is_goal), computes the percentile rank for each predicted probability, and then merges 
    these percentiles with the actual labels into a single DataFrame.

    Parameters:
    - pred_probs (np.ndarray): A 2D numpy array with probability estimates from a classifier. The 
      second column should contain the probabilities of the positive class.
    - y_val (pd.Series or pd.DataFrame): The true binary labels for the validation set.

    Returns:
    - df_percentile (pd.DataFrame): A DataFrame containing the goal probabilities, actual labels, 
      and the calculated percentile ranks.

    """
    #Create a df for shot probabilities
    df_probs = pd.DataFrame(pred_probs)
    df_probs = df_probs.rename(columns={0: "Not_Goal_prob", 1: "Goal_prob"})
    
    y_val = pd.DataFrame(y_val)
    y_val = y_val.rename(columns={0: "is_goal"})
    # Combining 'Goal Probability' and 'Is Goal' into one df. 
    df_probs = pd.concat([df_probs["Goal_prob"].reset_index(drop=True), y_val["is_goal"].reset_index(drop=True)],axis=1)
    
    # Computing and adding Percentile Column
    percentile_values=df_probs['Goal_prob'].rank(pct=True)
    df_probs['Percentile'] = percentile_values*100
    df_percentile = df_probs.copy()
    
    return df_percentile

def goal_rate(df_percentile):
    """
    Calculates the goal rate per percentile bin and returns a DataFrame with these rates.

    This function takes a DataFrame containing the percentile ranks of predicted probabilities 
    and the actual goal outcomes. It divides the data into bins based on percentile ranks and 
    calculates the rate of goals in each bin. The goal rate is defined as the number of goals 
    divided by the total number of shots in the bin, expressed as a percentage.

    Parameters:
    - df_percentile (pd.DataFrame): A DataFrame with two columns: 'Percentile' which contains 
      the percentile ranks, and 'is_goal' which contains the binary indicator of whether a shot 
      was a goal (1) or not (0).

    Returns:
    - goal_rate_df (pd.DataFrame): A DataFrame with each row representing a bin and two columns: 
      'Rate', the percentage of shots that were goals in each bin; and 'Percentile', the lower 
      bound of the percentile bin.
    """
    rate_list = []
    
    # Find total number of goals
    #total_goals = df_percentile['is_goal'].value_counts()[1]
   
    
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

def plot_goal_rates(goal_rate_df):
    """
    Plots the goal rates against shot probability model percentiles.

    This function takes a DataFrame containing goal rates and their corresponding percentile bins
    and plots the rates on the y-axis against the percentile bins on the x-axis. The plot is styled
    with grid lines, a light grey background, and custom tick marks. The x-axis is inverted so that
    the highest percentiles appear on the left. This visualization helps in understanding the
    relationship between the model's confidence in its predictions (as measured by the percentile
    rank of predicted probabilities) and the actual rate of goals scored.

    Parameters:
    - goal_rate_df (pd.DataFrame): A DataFrame with 'Rate' and 'Percentile' columns. 'Rate'
      should contain the goal rate for each bin, and 'Percentile' should contain the lower bound
      of the percentile bin.
    Returns:
    None
    """ 

    plt.figure(figsize=(8,6))
    
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
    
    #ax.legend(['Model 1'])
    
    plt.xlabel('Shot probability model percentile', fontsize=16)
    plt.title('Goal Rate', fontsize=16)
    plt.ylabel('Goals / (Shots+Goals)%', fontsize=16)
    plt.savefig('goal_rate_plot.png')
    
def plot_cumulative_goal_rates(df_percentile):
    """
    Plots the cumulative distribution of goals over the percentile ranks of predicted probabilities.

    The function filters the input DataFrame for actual goals and uses the empirical cumulative 
    distribution function (ECDF) to plot the proportion of total goals that are below each 
    percentile rank threshold. The x-axis represents the percentile ranks adjusted to a descending 
    order (i.e., higher percentiles on the left), and the y-axis represents the cumulative 
    proportion of goals.

    Parameters:
    - df_percentile (pd.DataFrame): A DataFrame containing a 'Percentile' column with the 
      percentile ranks and an 'is_goal' column indicating if the event was a goal (1) or not (0).

    Returns:
    None

    
    """    
    plt.figure(figsize=(8,6))
    df_precentile_only_goal = df_percentile[df_percentile['is_goal'] == 1]
    
    ax = sns.ecdfplot(data=df_precentile_only_goal, x=100 - df_precentile_only_goal.Percentile)
    
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
    ax.set_title(f"Cumulative % of Goals", fontsize=16)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.savefig('cumulative_goal_rate.png')
    ax.legend(['Logistic Regression'])
    
def plot_calibration_curve_prediction(y_val, pred_probs):
    """
    Plots a calibration curve for the predicted probabilities against the actual outcomes.

    The calibration curve, also known as a reliability diagram, shows the relationship between 
    predicted probabilities and the actual outcomes. It visualizes how well the predicted 
    probabilities of a classifier are calibrated. The function uses the `CalibrationDisplay` 
    from the scikit-learn library to generate the plot. The number of bins to discretize the 
    [0, 1] range into uniform bins for calibration is set to 50.

    Parameters:
    - y_val (pd.DataFrame or pd.Series): Actual binary labels for the validation set.
    - pred_probs (np.ndarray): Predicted probabilities for the positive class (second column of the array).

    Returns:
    None
    """
    y_val = pd.DataFrame(y_val)
    y_val = y_val.rename(columns={0: "is_goal"})

    plt.figure(figsize=(8,6))
    
    ax = CalibrationDisplay.from_predictions(y_val['is_goal'],pred_probs[:,1], n_bins=50)
   
    ax = plt.gca()
    ax.grid()
    ax.set_facecolor('0.95')
    ax.set_title(f"Calibration Curve", fontsize=16)
    plt.ylabel('Fraction of positives', fontsize=16)
    plt.xlabel('Mean predicted probability', fontsize=16)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.savefig('calibration_curve.png')
