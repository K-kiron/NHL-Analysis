# from comet_ml import Experiment
import numpy as np
import os
import sys


DATA_PATH = '../../IFT6758_Data/'
PROJECT_PATH = '../../Milestone2/'


sys.path.append(PROJECT_PATH)
from features.tidy_data import tidy_data
from visualizations.simple_visualization import *
    

    
def standardize_emptyNet(df: pd.DataFrame) -> int:
    """
    Standardize the emptyNet column

    Arguments:
        df: the dataframe of the shot data

    Return: 
        1 if the shot is an empty net, 0 otherwise
    """
    if df['emptyNet'] == True:
        return 1
    else:
        return 0
    

    
def is_goal(df: pd.DataFrame) -> int:
    """
    Standardize the is_goal column

    Arguments:
        df: the dataframe of the shot data

    Return:
        1 if the shot is a goal, 0 otherwise
    """
    if df['eventType'] == 'Goal':
        return 1
    else:
        return 0
    


def generate_train_set(DATA_PATH) -> pd.DataFrame:
    """
    Generate the training set(2016-2019) from the raw data

    Arguments:
        DATA_PATH: the path of the raw data

    Return:
        the training set as a DataFrame
    """

    train_df = year_integration(DATA_PATH, 2016)
    train_df = pd.concat([train_df, year_integration(DATA_PATH, 2017)], ignore_index=True)
    train_df = pd.concat([train_df, year_integration(DATA_PATH, 2018)], ignore_index=True)
    train_df = pd.concat([train_df, year_integration(DATA_PATH, 2019)], ignore_index=True)

    train_df['emptyNet'] = train_df.apply(standardize_emptyNet, axis=1)
    train_df['is_goal'] = train_df.apply(is_goal, axis=1)

    # shuffle the dataframe
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    return train_df




def init_distance_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize the distance bins

    Arguments:
        df: the dataframe of the shot data

    Return: 
        the dataframe with distance bins
    """
    max_distance = df['shotDistance'].max()
    min_distance = 0

    # divide the distance into 20 bins
    bins = np.linspace(min_distance, max_distance, 20)

    df['distance_bin'] = pd.cut(df['shotDistance'], bins=bins)
    return df



def init_angle_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize the angle bins

    Arguments:
        df: the dataframe of the shot data

    Return:
        the dataframe with angle bins
    """
    max_angle = 90
    min_angle = 0

    # divide the distance into 20 bins
    bins = np.linspace(min_angle, max_angle, 20)
        
    df['angle_bin'] = pd.cut(df['shotAngle'], bins=bins)
    return df



def bin_by_distance(df: pd.DataFrame):
    """
    plot the histogram of shot counts by distance

    Arguments:
        df: the dataframe of the shot data

    Return:
        None
    """

    bins = np.linspace(0, df['shotDistance'].max(), 20)
    
    df_goal = df[df['is_goal'] == 1]
    df_no_goal = df[df['is_goal'] == 0]

    plt.figure(figsize=(8, 5))
    plt.hist(df_no_goal['shotDistance'], bins=bins, alpha=0.5, label='no-goal', edgecolor='black', linewidth=1.2)
    plt.hist(df_goal['shotDistance'], bins=bins, alpha=0.5, label='goal', edgecolor='black', linewidth=1.2)
    plt.xlabel('Distance from the goalpost')
    plt.ylabel('Shot counts')
    plt.title('Shot counts by distance')
    plt.legend(loc='upper right')
    plt.show()



def bin_by_angle(df: pd.DataFrame):
    """
    plot the histogram of shot counts by angle

    Arguments:
        df: the dataframe of the shot data

    Return:
        None
    """

    bins = np.linspace(0, 90, 20)
        
    df_goal = df[df['is_goal'] == 1]
    df_no_goal = df[df['is_goal'] == 0]
    
    plt.figure(figsize=(8, 5))
    plt.hist(df_no_goal['shotAngle'], bins=bins, alpha=0.5, label='no-goal', edgecolor='black', linewidth=1.2)
    plt.hist(df_goal['shotAngle'], bins=bins, alpha=0.5, label='goal', edgecolor='black', linewidth=1.2)
    plt.xlabel('Shot angle')
    plt.ylabel('Shot counts')
    plt.title('Shot counts by angle')
    plt.legend(loc='upper right')
    plt.show()


def joint_plot(df: pd.DataFrame):
    """
    plot the joint plot of distance and angle

    Arguments:
        df: the dataframe of the shot data

    Return:
        None
    """
    plt.figure(figsize=(8, 5))
    sns.jointplot(x='shotDistance', y='shotAngle', data=df, kind='hist', bins=20)
    plt.xlabel('Distance from the goalpost')
    plt.ylabel('Shot angle')
    plt.show()



def prob_by_distance(df: pd.DataFrame):
    """
    Divide the distance into 20 intervals and plot the goal rate by distance

    Arguments:
        df: the dataframe of the shot data

    Return:
        None
    """

    df = init_distance_bins(df)
    
    df_goal = df[df['is_goal'] == 1]
    df_goal_count = df_goal.groupby(['distance_bin']).count()['is_goal']
    df_total_count = df.groupby(['distance_bin']).count()['is_goal']
    df_goal_prob = df_goal_count / df_total_count

    plt.figure(figsize=(8, 5))
    df_goal_prob.plot(kind='bar', x='distance_bin', y='is_goal')
    plt.xlabel('Distance from the goalpost')
    plt.ylabel('Goal rate')
    plt.title('Goal rate by distance')
    plt.show()



def prob_by_angle(df: pd.DataFrame):
    """
    Divide the angle into 20 intervals and plot the goal rate by angle

    Arguments:
        df: the dataframe of the shot data

    Return:
        None
    """

    df = init_angle_bins(df)
    
    df_goal = df[df['is_goal'] == 1]
    df_goal_count = df_goal.groupby(['angle_bin']).count()['is_goal']
    df_total_count = df.groupby(['angle_bin']).count()['is_goal']
    df_goal_prob = df_goal_count / df_total_count

    plt.figure(figsize=(8, 5))
    df_goal_prob.plot(kind='bar', x='angle_bin', y='is_goal')
    plt.xlabel('Shot angle')
    plt.ylabel('Goal rate')
    plt.title('Goal rate by angle')
    plt.show()



def check_emptyNet(df: pd.DataFrame):
    """
    Split the data into empty net and non-empty net and plot the goal rate by distance

    Arguments:
        df: the dataframe of the shot data

    Return:
        None
    """

    bins = np.linspace(0, df['shotDistance'].max(), 20)

    df_goal = df[df['is_goal'] == 1]
    df_emptyNet = df_goal[df_goal['emptyNet'] == 1]
    df_not_emptyNet = df_goal[df_goal['emptyNet'] == 0]

    plt.figure(figsize=(8, 5))
    plt.hist(df_not_emptyNet['shotDistance'], bins=bins, alpha=0.5, label='non-empty net', edgecolor='black', linewidth=1.2, color='grey')
    plt.hist(df_emptyNet['shotDistance'], bins=bins, alpha=0.5, label='empty net', edgecolor='black', linewidth=1.2, color='lightgreen')
    plt.xlabel('Distance from the goalpost')
    plt.ylabel('Shot counts')
    plt.title('Shot counts by distance')
    plt.legend(loc='upper right')
    plt.show()
