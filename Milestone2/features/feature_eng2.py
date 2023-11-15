import json
import os
import glob
import pandas as pd
import numpy as np
import re
from tidy_data import *

def game_data(path, year, season, game_id):
    with open(os.path.join(f'{path}/{year}/{season}/', f'{game_id}.json'), 'r') as file:
        game_data = json.load(file)
    return game_data

def binary2cumulative(lst):
    nlst = []
    count = 0
    for i in lst:
        if i == 1:
            count += 1
        else:
            count = 0  
        nlst.append(count)
    return nlst

def power_play(df, select_period):
    '''
    Returns arrays documenting time since power play started for all teams, home team, and away team for a chosen period
    Returns arrays documenting the number of player on the field as a function of time for a chosen period

    Parameters:
        df (DataFrame): The DataFrame outputed by game_data
        select_period (int): period for which the outputs are returned

    Returns:
        binary2cumulative(timer): array where the entry is the time since the power play started for penalties
                                        given to ALL teams combined, where the index is the time in seconds
        binary2cumulative(timer_home): array where the entry is the time since the power play started for penalties
                                        given to the home team, where the index is the time in seconds
        binary2cumulative(timer_away): array where the entry is the time since the power play started for penalties
                                        given to the away team, where the index is the time in seconds
        players_home: array where the entry is the number of home players on the field,
                        where the index is the time in seconds
        players_away: array where the entry is the number of away players on the field,
                        where the index is the time in seconds
    '''
        
    idx = df['liveData']['plays']['penaltyPlays']
    home_team = df['liveData']['boxscore']['teams']['home']['team']['triCode']
    away_team = df['liveData']['boxscore']['teams']['away']['team']['triCode']

    TotalT = 60*20
    
    timer = np.zeros(TotalT)
    timer_away = np.zeros(TotalT)
    players_away = 5*np.ones(TotalT)

    timer_home = np.zeros(TotalT)
    players_home = 5*np.ones(TotalT)

    time_elapsed = 0
    period_count = 0
    for i in idx:
        penalty_event = df['liveData']['plays']['allPlays'][i]
        period = penalty_event['about']['period']
        if select_period == period:
            periodTimeRemaining =  60*int(penalty_event['about']['periodTimeRemaining'].split(':')[0]) + int(penalty_event['about']['periodTimeRemaining'].split(':')[1])
            penalty_t0 = 60*int(penalty_event['about']['periodTime'].split(':')[0]) + int(penalty_event['about']['periodTime'].split(':')[1])
            penalty_length = min(60*int(penalty_event['result']['penaltyMinutes']) , periodTimeRemaining)
            penalty_t1 = penalty_length + penalty_t0
            periodLength_s = penalty_t0+periodTimeRemaining
            team = penalty_event['team']['triCode']

            if team == home_team:
                for i in range(time_elapsed+penalty_t0, time_elapsed+penalty_t1):
                    timer_home[i] = 1
                    timer[i] = 1
                    players_home[i] = players_home[i]-1
            elif team == away_team:
                for i in range(time_elapsed+penalty_t0, time_elapsed+penalty_t1):
                    timer_away[i] = 1
                    timer[i] = 1
                    players_away[i] = players_away[i]-1
      
    return binary2cumulative(timer), binary2cumulative(timer_home), binary2cumulative(timer_away), players_home, players_away

def is_goal(df):
    if df['eventType'] == 'Goal':
        return 1
    else:
        return 0

def feature_eng2_raw(DATA_PATH, year, season, game_id):
    df = tidy_data(DATA_PATH, year, season, game_id)
    if len(df) == 0:
        return
    df_eng2 = df
    df_eng2[['minutes', 'seconds']] = df_eng2['periodTime'].str.split(':', expand=True)
    df_eng2['gameSeconds'] = df['period']*(df_eng2['minutes'].astype(int) * 60 + df_eng2['seconds'].astype(int))
    df_eng2 = df_eng2.drop(columns=['minutes', 'seconds'])

    df_copy = df
    new_df = df_copy.shift(fill_value=np.nan)
    new_df.columns = df.columns

    new_df.iloc[0, :] = np.nan
    new_df.iloc[:, 0] = np.nan

    df_copy = new_df
    df_eng2['LastEventType'] = df_copy['eventType']
    df_eng2['Last_x_coordinate'] = df_copy['x_coordinate']
    df_eng2['Last_y_coordinate'] = df_copy['y_coordinate']
    df_eng2['Last_gameSeconds'] = df_copy['gameSeconds']
    df_eng2['Last_period'] = df_copy['period']
    df_eng2['DistanceLastEvent'] = np.sqrt((df_eng2['Last_x_coordinate']-df_eng2['x_coordinate'])**2+(df_eng2['Last_y_coordinate']-df_eng2['y_coordinate'])**2)
    df_eng2['Rebound'] = df_eng2['LastEventType'] == 'Shot'
    df_eng2['LastShotAngle'] = df_copy['shotAngle']
    df_eng2['changeShotAngle'] = df_eng2['LastShotAngle']+df_eng2['shotAngle']
    df_eng2['timeFromLastEvent'] = df_eng2['gameSeconds']-df_eng2['Last_gameSeconds']
    df_eng2['speed'] = df_eng2['DistanceLastEvent']/df_eng2['timeFromLastEvent']
    
    dfg = game_data(DATA_PATH, year, season, game_id)
    period_lst = df_eng2['period'].tolist()
    time_since_pp = np.array([[]])
    no_players_home = np.array([[]])
    no_players_away = np.array([[]])
    for select_period in range(min(period_lst),max(period_lst)+1):
        selected_idx = np.where(np.array(period_lst) == select_period)
        selected_times = df_eng2['gameSeconds'].iloc[selected_idx]
        selected_times = np.array(selected_times%(60*20))
        time_since_pp = np.concatenate((time_since_pp, np.array(power_play(dfg, select_period)[0])[selected_times]), axis=None)
        no_players_home = np.concatenate((no_players_home, np.array(power_play(dfg, select_period)[3])[selected_times]), axis=None)
        no_players_away = np.concatenate((no_players_away, np.array(power_play(dfg, select_period)[4])[selected_times]), axis=None)

    df_eng2['time_since_pp'] = time_since_pp
    df_eng2['no_players_home'] = no_players_home
    df_eng2['no_players_away'] = no_players_away
    df_eng2['is_goal'] = df_eng2.apply(is_goal, axis=1)
    
    
    return df_eng2

json_regex = re.compile(r"(.*)\.json")

def season_integration_eng2(path, year, season) -> pd.DataFrame:
    df = pd.DataFrame()
    season_path = os.path.join(path, str(year), season)
    i = 0

    for file_path in os.listdir(season_path):
        match = re.match(json_regex, file_path)
        if match:
            game_id = match.group(1)
            game_df = feature_eng2_raw(path, year, season, game_id)
            df = pd.concat([df, game_df], ignore_index=True)
    return df


def feature_eng2(path, year,season) -> pd.DataFrame:
    df = pd.DataFrame()
    #for season in ['regular', 'playoffs']:
    #    season_df = season_integration_eng2(path, year, season)
    #    df = pd.concat([df, season_df], ignore_index=True)
    #return df
    season_df = season_integration_eng2(path, year, season)
    df = pd.concat([df, season_df], ignore_index=True)
    return df

def feature_eng2_cleaned(path, year, season) -> pd.DataFrame:
    df = pd.DataFrame()
    #for season in ['regular', 'playoffs']:
    #    season_df = season_integration_eng2(path, year, season)
    #    df = pd.concat([df, season_df], ignore_index=True)
    #return df[['gameSeconds','period','x_coordinate','y_coordinate','shotDistance','shotAngle','shotType','LastEventType','Last_x_coordinate','Last_y_coordinate','timeFromLastEvent','DistanceLastEvent','Rebound','changeShotAngle','speed','time_since_pp','no_players_home','no_players_away', 'is_goal']]
    season_df = season_integration_eng2(path, year, season)
    df = pd.concat([df, season_df], ignore_index=True)
    return df[['gameSeconds','period','x_coordinate','y_coordinate','shotDistance','shotAngle','shotType','LastEventType','Last_x_coordinate','Last_y_coordinate','timeFromLastEvent','DistanceLastEvent','Rebound','changeShotAngle','speed','time_since_pp','no_players_home','no_players_away', 'is_goal']]

def get_train_data(DATA_PATH):
    data = feature_eng2_cleaned(DATA_PATH, 2016)
    data = pd.concat([data, feature_eng2_cleaned(DATA_PATH, 2017)], ignore_index=True)
    data = pd.concat([data, feature_eng2_cleaned(DATA_PATH, 2018)], ignore_index=True)
    data = pd.concat([data, feature_eng2_cleaned(DATA_PATH, 2019)], ignore_index=True)

    data.to_csv(DATA_PATH + '/clean_train_data.csv')

def get_test_data(DATA_PATH):
    #data = feature_eng2_cleaned(DATA_PATH, 2020, 'regular')
    #data.to_csv(DATA_PATH + '/clean_test_data_regular.csv')

    data = feature_eng2(DATA_PATH, 2020,'playoffs')
    data.to_csv(DATA_PATH + '/clean_test_data_playoff.csv')

def get_full_test_data(DATA_PATH):
    data = feature_eng2(DATA_PATH, 2020)

    data.to_csv(DATA_PATH + '/full_test_data.csv')

def get_full_train_data(DATA_PATH):
    data = feature_eng2(DATA_PATH, 2016)
    data = pd.concat([data, feature_eng2(DATA_PATH, 2017)], ignore_index=True)
    data = pd.concat([data, feature_eng2(DATA_PATH, 2018)], ignore_index=True)
    data = pd.concat([data, feature_eng2(DATA_PATH, 2019)], ignore_index=True)

    data.to_csv(DATA_PATH + '/full_train_data.csv')

if __name__ == "__main__":
    dataset = input('Enter train for parsing train data and test for parsing test data: ')
    subset_or_full = input('Enter subset for only feature engineering 2 data and full for all data: ')

    dataset = dataset.strip()
    subset_or_full = subset_or_full.strip()
    assert dataset in ['train', 'test']
    assert subset_or_full in ['subset', 'full']

    if dataset == 'train':
        if subset_or_full == 'subset':
            get_train_data('../../IFT6758_Data')
        else:
            get_full_train_data('../../IFT6758_Data')
    else:
        if subset_or_full == 'subset':
            get_test_data('../../IFT6758_Data')
        else:
            get_full_test_data('../../IFT6758_Data')
