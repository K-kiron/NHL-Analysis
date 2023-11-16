import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import roc_curve, auc
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Read in data and assign X and y
data = pd.read_csv('../../../IFT6758_Data/train_data.csv', index_col=0)
X = data[['shotDistance', 'shotAngle' ]]
X = X.rename({'shotDistance': 'distanceFromNet', 'shotAngle': 'angleFromNet'}, axis=1)
X.interpolate(method='linear', inplace=True)
y = data[['is_goal']]
# Drop rows with NaN values from both X and y
X = X.dropna()
y = y.loc[X.index]

def Log_reg(X, y, feature_list):
    """
    Trains and validates a logistic regression model using a specified list of features.
    
    This function splits the input data into training and validation sets, 
    trains a logistic regression model on the training set, and calculates 
    the accuracy of the model on the validation set. It returns the validation 
    set, the predicted values, the model's accuracy, and the probability 
    estimates for the validation set.

    Parameters:
    - X (pd.DataFrame): A DataFrame containing the features of the dataset.
    - y (pd.Series or pd.DataFrame): The target variable associated with the dataset.
    - feature_list (list): A list of strings representing the names of the features to be used for training the logistic regression model.

    Returns:
    - X_val (pd.DataFrame): The features of the validation set.
    - y_val (pd.Series or pd.DataFrame): The true target values for the validation set.
    - y_pred (np.ndarray): The predictions made by the model for the validation set.
    - accuracy (float): The accuracy score of the model on the validation set.
    - pred_probs (np.ndarray): The probability estimates for the validation set.

    Note:
    - The function sets the `random_state` parameter to 42 to ensure reproducibility.
    - The function prints the model's accuracy on the validation set.
    
    """

    #print(X[feature_list])
    X_train,X_val,y_train,y_val = train_test_split(X[feature_list], y, test_size=0.2, random_state=42)

    # Logistic regression model fitting
    clf = LogisticRegression()
    y_train = y_train.values.ravel()
    clf.fit(X_train, y_train)

    # Predict on validation set
    y_pred = clf.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, y_pred)

    #X_test_pred_proba = clf.predict_proba(X_test)
    pred_probs = clf.predict_proba(X_val)

    return X_val, y_val, y_pred, accuracy,  pred_probs

def plot_roc_all_feat(X, y):

    """
    Plots ROC curves for logistic regression models trained on different feature sets.

    This function iterates over a list of feature sets, trains a logistic regression model for each,
    and plots the ROC curve on the same graph for comparison. The ROC curves are color-coded for 
    distinction. It also calculates and displays the AUC score for each model. Additionally, this 
    function plots a ROC curve for a random baseline model.

    Parameters:
    - X (pd.DataFrame): The feature data used for training the models.
    - y (pd.Series or pd.DataFrame): The target variable for the models.

    Returns:
    None
    Outputs:
    A plot is displayed showing ROC curves for each feature set and a random baseline. The plot 
    is saved as '3a_ROC_curves.png' in the current working directory.
    """

    fig = plt.figure(figsize=(12,10))

    feature_list = (['distanceFromNet'], ['angleFromNet'], ['distanceFromNet', 'angleFromNet']  )
    feature_color_list = ['red', 'blue', 'green']
    plot_label_list = ['Distance from Net', 'Angle from Net', 'Distance and Angle from Net']

    for i, feature in enumerate(feature_list):
        X_val, y_val, y_pred, accuracy,  pred_probs = Log_reg(X, y, feature)
        print(f'Accuracy score is {accuracy}')

        plot_color = feature_color_list[i]
        plot_label = plot_label_list[i]

        probs_isgoal = pred_probs[:,1]
        fpr, tpr, _ = roc_curve(y_val,probs_isgoal)
        roc_auc = auc(fpr,tpr)

        plt.plot(fpr, tpr, color = plot_color, label = f'{plot_label} '+'AUC = %0.2f' % roc_auc, lw=2)

    #Random Baseline
    baseline_is_goal = np.random.uniform(0,1,probs_isgoal.shape[0])
    plot_color = 'Magenta'
    plot_label = 'Random Baseline'
    fpr, tpr, _ = roc_curve(y_val,baseline_is_goal)
    roc_auc = auc(fpr,tpr)
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
    plt.savefig(f'3a_ROC_curves.png')
    plt.show()

plot_roc_all_feat(X,y)

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

    # Combining 'Goal Probability' and 'is-goal' into one df.
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

def plot_goal_rate_all_feat(X, y):
    """
    Plots goal rates against shot probability model percentiles for different feature sets.

    For each set of features provided, this function trains a logistic regression model,
    calculates the percentile of predicted probabilities, computes goal rates, and plots
    these rates. It generates a single figure with multiple lines representing the goal
    rates as a function of model confidence percentiles for each feature set. Additionally,
    it computes and plots a random baseline for comparison.

    Parameters:
    - X (pd.DataFrame): The feature data used for training the models.
    - y (pd.Series or pd.DataFrame): The target variable for the models.

    Returns:
    None
    
    Outputs:
    A plot is displayed showing goal rates for each feature set and a random baseline. The plot 
    is saved as '3b_goal_rates.png' in the current working directory.
    """

    fig = plt.figure(figsize=(12,10))

    feature_list = (['distanceFromNet'], ['angleFromNet'], ['distanceFromNet', 'angleFromNet']  )
    feature_color_list = ['red', 'blue', 'green']
    plot_label_list = ['Distance from Net', 'Angle from Net', 'Distance and Angle from Net']

    #if model_name == 'LR':
    for i, feature in enumerate(feature_list):

        X_val, y_val, y_pred, accuracy,  pred_probs = Log_reg(X, y, feature)
        print(f'Accuracy score is {accuracy}')

        plot_color = feature_color_list[i]
        plot_label = plot_label_list[i]

        df_percentile =  calc_percentile(pred_probs, y_val)
        goal_rate_df = goal_rate(df_percentile)
        goal_rate_x = goal_rate_df['Percentile']
        goal_rate_y = goal_rate_df['Rate']
        plt.plot(goal_rate_x,goal_rate_y, color = plot_color, label = f'{plot_label}' )


    #Random Baseline
    probs_isgoal = pred_probs[:,1]
    baseline_is_goal = np.random.uniform(0,1,probs_isgoal.shape[0])
    no_baseline_goal = np.array([(1-i) for i in baseline_is_goal])
    random_probs = np.column_stack((baseline_is_goal, no_baseline_goal))
    df_percentile =  calc_percentile(random_probs, y_val)
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
    plt.ylabel('Goals / (Shots)%', fontsize=16)
    plt.legend(loc=2,prop={'size': 16})
    plt.tight_layout()
    plt.savefig(f'3b_goal_rates.png')
    plt.show()

plot_goal_rate_all_feat(X,y)

def plot_cumulative_rate_all_feat(X, y):
    """
    Plots the empirical cumulative distribution function (ECDF) of goals for logistic regression models
    trained on different sets of features, including a comparison to a random baseline.

    For each specified feature set, the function trains a logistic regression model and computes the
    percentile of the predicted probabilities. It then filters for actual goal outcomes and plots the
    ECDF, showing the proportion of goals below each percentile rank threshold. The plots are
    color-coded for each feature set for easy comparison. A random baseline ECDF is also plotted.

    Parameters:
    - X (pd.DataFrame): The feature data used for training the models.
    - y (pd.Series or pd.DataFrame): The target variable for the models.

    Returns:
    None

    Outputs:
    A plot is displayed showing the ECDF of goals for each feature set and a random baseline. The plot
    is saved as '3c_goal_proportions.png' in the current working directory.
    """
    fig = plt.figure(figsize=(12,10))

    feature_list = (['distanceFromNet'], ['angleFromNet'], ['distanceFromNet', 'angleFromNet']  )
    feature_color_list = ['red', 'blue', 'green']
    plot_label_list = ['Distance from Net', 'Angle from Net', 'Distance and Angle from Net']

    #if model_name == 'LR':
    for i, feature in enumerate(feature_list):

        X_val, y_val, y_pred, accuracy,  pred_probs = Log_reg(X, y, feature)
        print(f'Accuracy score is {accuracy}')

        plot_color = feature_color_list[i]
        plot_label = plot_label_list[i]

        df_percentile =  calc_percentile(pred_probs, y_val)
        df_precentile_only_goal = df_percentile[df_percentile['is_goal'] == 1]
        ax = sns.ecdfplot(data=df_precentile_only_goal, x=100 - df_precentile_only_goal.Percentile,
                              color=plot_color)



    #Random Baseline
    probs_isgoal = pred_probs[:,1]
    baseline_is_goal = np.random.uniform(0,1,probs_isgoal.shape[0])
    no_baseline_goal = np.array([(1-i) for i in baseline_is_goal])
    random_probs = np.column_stack((baseline_is_goal, no_baseline_goal))
    df_percentile =  calc_percentile(random_probs, y_val)
    df_precentile_only_goal = df_percentile[df_percentile['is_goal'] == 1]

    plot_color = 'Magenta'
    plot_label = 'Random Baseline'
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

    plot_label_list.append('Random Baseline')
    plt.legend(labels=plot_label_list, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'3c_goal_proportions.png')
    plt.show()

plot_cumulative_rate_all_feat(X, y)

def plot_calibration_all_feat(X, y):
    """
    Plots calibration curves for logistic regression models trained on different feature sets, 
    including a calibration curve for a random baseline.

    For each specified feature set, the function trains a logistic regression model, obtains 
    predicted probabilities, and uses these to plot a calibration curve, which shows how well the 
    predicted probabilities are calibrated. A calibration curve for a random baseline is also plotted 
    for comparison. The subplots are arranged in a grid where each row corresponds to a feature set.

    Parameters:
    - X (pd.DataFrame): The feature data used for training the models.
    - y (pd.Series or pd.DataFrame): The target variable for the models.

    Returns:
    None

    Outputs:
    A plot is displayed showing calibration curves for each feature set and a random baseline. The plot 
    is saved as '3d_calibration_plots.png' in the current working directory.
    """
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 3)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])

    feature_list = [['distanceFromNet'], ['angleFromNet'], ['distanceFromNet', 'angleFromNet']  ]
    feature_color_list = ['red', 'blue', 'green']
    plot_label_list = ['Distance from Net', 'Angle from Net', 'Distance and Angle from Net']

    feature_list.append('RandomBaseline')

    #if model_name == 'LR':
    for i, feature in enumerate(feature_list):

        if feature != 'RandomBaseline':

            X_val, y_val, y_pred, accuracy,  pred_probs = Log_reg(X, y, feature)
            print(f'Accuracy score is {accuracy}')

            y_val_is_goal = y_val['is_goal']
            pred_probs_is_goal = pred_probs[:,1]

            plot_color = feature_color_list[i]
            plot_label = plot_label_list[i]

        else:
            random_goal_prob = np.random.uniform(0, 1, len(y_val))

            y_val_is_goal = y_val['is_goal']
            pred_probs_is_goal = random_goal_prob.copy()

            plot_color = 'magenta'
            plot_label = 'Random Baseline'


        ax_display = CalibrationDisplay.from_predictions(y_val_is_goal,pred_probs_is_goal, n_bins=50,
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
    plt.savefig(f'3d_calibration_plots.png')
    plt.show()

plot_calibration_all_feat(X, y)
