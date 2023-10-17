import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re
from IPython.display import display

DATA_PATH = '../data/IFT6758_Data'
PROJECT_PATH = '../../Milestone1/'

sys.path.append(PROJECT_PATH)
from features.tidy_data import tidy_data



json_regex = re.compile(r"(.*)\.json")

def season_integration(path, year, season) -> pd.DataFrame:
    df = pd.DataFrame()
    season_path = os.path.join(path, str(year), season)
    i = 0

    for file_path in os.listdir(season_path):
        match = re.match(json_regex, file_path)
        if match:
            game_id = match.group(1)
            game_df = tidy_data(path, year, season, game_id)
            df = pd.concat([df, game_df], ignore_index=True)
    return df



def year_integration(path, year) -> pd.DataFrame:
    df = pd.DataFrame()
    for season in ['regular', 'playoffs']:
        season_df = season_integration(path, year, season)
        df = pd.concat([df, season_df], ignore_index=True)
    return df

def calculate_shot_distance(df):
    if df['goal_location'] == 'Left':
        return np.sqrt((df['x_coordinate'] - (-89))**2 + (df['y_coordinate'] - 0)**2)
    elif df['goal_location'] == 'Right':
        return np.sqrt((df['x_coordinate'] - 89)**2 + (df['y_coordinate'] - 0)**2)
    

max_distance = np.sqrt((89*2)**2 + 0**2)
min_distance = 0
bins = np.linspace(min_distance, max_distance, 20)


def calculate_probability_distance(start_year, end_year):
    num_year = end_year - start_year + 1
    for i in range(num_year):
        year = start_year + i
        df = year_integration(DATA_PATH, year)
        df['shotDistance'] = df.apply(calculate_shot_distance, axis=1)
        df['shotDistanceBin'] = pd.cut(df['shotDistance'], bins=bins)
        df_shot = df[df['eventType'] == 'Shot'].reset_index(drop=True)
        df_goal = df[df['eventType'] == 'Goal'].reset_index(drop=True)
        df_goal_count = df_goal.groupby('shotDistanceBin').count()['eventType']
        df_total_count = df.groupby('shotDistanceBin').count()['eventType']
        df_goal_prob = df_goal_count / df_total_count
        plt.figure(figsize=(16, 5))
        plt.suptitle('Probability of Goal at different Shot Distances in Season {}'.format(year))
        plt.subplot(1, 2, 1)
        df_goal_prob.plot(kind='bar')
        plt.xlabel('Shot Distance')
        plt.ylabel('Probability of Goal')
        plt.subplot(1, 2, 2)
        df_goal_prob.plot()
        plt.xticks(rotation=90)
        plt.xlabel('Shot Distance')
        plt.ylabel('Probability of Goal')
        plt.show()

def calculate_probability_distance_shottypes(year):
    df = year_integration(DATA_PATH, year)
    df['shotDistance'] = df.apply(calculate_shot_distance, axis=1)
    df['shotDistanceBin'] = pd.cut(df['shotDistance'], bins=bins)
    df_goal = df[df['eventType'] == 'Goal'].reset_index(drop=True)
    df_goal_st_count = df_goal.groupby(['shotDistanceBin', 'shotType']).count()['eventType']
    df_total_st_count = df.groupby(['shotDistanceBin', 'shotType']).count()['eventType']
    df_goal_st_prob = df_goal_st_count / df_total_st_count
    df_goal_st_prob = df_goal_st_prob.unstack()
    df_goal_st_prob.fillna(0, inplace=True)
    sns.heatmap(df_goal_st_prob, annot=True, fmt='.2f', cmap='Purples')
    plt.title('Probability of Goal of each Shot Type at different Shot Distances in Season {}'.format(year))
    plt.xlabel('Shot Type')
    plt.ylabel('Shot Distance')
    plt.show()