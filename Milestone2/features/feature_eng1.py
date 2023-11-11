# from comet_ml import Experiment
import numpy as np
import os
import sys


DATA_PATH = '../../IFT6758_Data/'
PROJECT_PATH = '../../Milestone2/'


sys.path.append(PROJECT_PATH)
from features.tidy_data import tidy_data
from visualizations.simple_visualization import *




def calculate_shot_angle(df):
    if df['goal_location'] == 'Left':
        # calculate angle to (-89, 0)
        return np.degrees(np.arctan2(np.abs(df['y_coordinate']), np.abs(df['x_coordinate'] + 89)))
    elif df['goal_location'] == 'Right':
        # calculate angle to (89, 0)
        return np.degrees(np.arctan2(np.abs(df['y_coordinate']), np.abs(df['x_coordinate'] - 89)))
    

    
def standardize_emptyNet(df):
    if df['emptyNet'] == True:
        return 1
    else:
        return 0
    

    
def is_goal(df):
    if df['eventType'] == 'Goal':
        return 1
    else:
        return 0
    

    
def generate_train_set(DATA_PATH):

    train_df = year_integration(DATA_PATH, 2016)
    train_df = pd.concat([train_df, year_integration(DATA_PATH, 2017)], ignore_index=True)
    train_df = pd.concat([train_df, year_integration(DATA_PATH, 2018)], ignore_index=True)
    train_df = pd.concat([train_df, year_integration(DATA_PATH, 2019)], ignore_index=True)

    train_df['shot_angle'] = train_df.apply(calculate_shot_angle, axis=1)
    train_df['shot_distance'] = train_df.apply(calculate_shot_distance, axis=1)
    train_df['emptyNet'] = train_df.apply(standardize_emptyNet, axis=1)
    train_df['is_goal'] = train_df.apply(is_goal, axis=1)

    # shuffle the dataframe
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    return train_df



def init_distance_bins(df):
    max_distance = df['shot_distance'].max()
    min_distance = 0

    # divide the distance into 20 bins
    bins = np.linspace(min_distance, max_distance, 20)

    df['distance_bin'] = pd.cut(df['shot_distance'], bins=bins)
    return df



def init_angle_bins(df):
    max_angle = 90
    min_angle = 0

    # divide the distance into 20 bins
    bins = np.linspace(min_angle, max_angle, 20)
        
    df['angle_bin'] = pd.cut(df['shot_angle'], bins=bins)
    return df



def bin_by_distance(df):
    
    df_goal = df[df['is_goal'] == 1]
    df_no_goal = df[df['is_goal'] == 0]

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df_goal['shot_distance'], bins=20)
    plt.xlabel('Distance from the gate')
    plt.ylabel('Number of goals')
    plt.title('Number of goals by distance')
    plt.subplot(1, 2, 2)
    plt.hist(df_no_goal['shot_distance'], bins=20)
    plt.xlabel('Distance from the gate')
    plt.ylabel('Number of no-goals')
    plt.title('Number of no-goals by distance')
    plt.show()



def bin_by_angle(df):
        
    df_goal = df[df['is_goal'] == 1]
    df_no_goal = df[df['is_goal'] == 0]
    
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df_goal['shot_angle'], bins=20)
    plt.xlabel('Shot angle')
    plt.ylabel('Number of goals')
    plt.title('Number of goals by angle')
    plt.subplot(1, 2, 2)
    plt.hist(df_no_goal['shot_angle'], bins=20)
    plt.xlabel('Shot angle')
    plt.ylabel('Number of no-goals')
    plt.title('Number of no-goals by angle')
    plt.show()


def joint_plot(df):
    plt.figure(figsize=(8, 5))
    sns.jointplot(x='shot_distance', y='shot_angle', data=df, kind='hist', bins=20)
    plt.xlabel('Distance from the gate')
    plt.ylabel('Shot angle')
    plt.show()



def prob_by_distance(df):

    df = init_distance_bins(df)
    
    df_goal = df[df['is_goal'] == 1]
    df_goal_count = df_goal.groupby(['distance_bin']).count()['is_goal']
    df_total_count = df.groupby(['distance_bin']).count()['is_goal']
    df_goal_prob = df_goal_count / df_total_count

    plt.figure(figsize=(8, 5))
    df_goal_prob.plot(kind='bar', x='distance_bin', y='is_goal')
    plt.xlabel('Distance from the gate')
    plt.ylabel('Goal rate')
    plt.title('Goal rate by distance')
    plt.show()



def prob_by_angle(df):

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



def check_emptyNet(df):

    # df = init_distance_bins(df)

    df_goal = df[df['is_goal'] == 1]
    df_no_goal = df[df['is_goal'] == 0]
    # display(df_no_goal)
    df_emptyNet = df_goal[df_goal['emptyNet'] == 1]
    df_emptyNet_no = df_no_goal[df_no_goal['emptyNet'] == 1]
    # display(df_emptyNet_no)
    df_not_emptyNet = df_goal[df_goal['emptyNet'] == 0]
    df_not_emptyNet_no = df_no_goal[df_no_goal['emptyNet'] == 0]

    # df_emptyNet_count = df_emptyNet.groupby(['distance_bin']).count()['is_goal']
    # df_not_emptyNet_count = df_not_emptyNet.groupby(['distance_bin']).count()['is_goal']

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df_emptyNet['shot_distance'], bins=20)
    plt.xlabel('Distance from the gate')
    plt.ylabel('Number of goals')
    plt.title('Number of goals by distance (empty net)')
    plt.subplot(1, 2, 2)
    plt.hist(df_emptyNet_no['shot_distance'], bins=20)
    plt.xlabel('Distance from the gate')
    plt.ylabel('Number of no-goals')
    plt.title('Number of no-goals by distance (empty net)')

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df_not_emptyNet['shot_distance'], bins=20)
    plt.xlabel('Distance from the gate')
    plt.ylabel('Number of goals')
    plt.title('Number of goals by distance (non-empty net)')
    plt.subplot(1, 2, 2)
    plt.hist(df_not_emptyNet_no['shot_distance'], bins=20)
    plt.xlabel('Distance from the gate')
    plt.ylabel('Number of no-goals')
    plt.title('Number of no-goals by distance (non-empty net)')
    plt.show()
