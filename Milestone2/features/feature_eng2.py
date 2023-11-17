import json
import os
import glob
import pandas as pd
import numpy as np
import re
import sys

DATA_PATH = '../../IFT6758_Data/'
PROJECT_PATH = '../../Milestone2/'


sys.path.append(PROJECT_PATH)
from features.tidy_data import tidy_data

def game_data(path: str, year: int, season: str, game_id: int) -> pd.DataFrame:
    """
    Arguments:
        path (str): DATA_PATH
        year (int): year in which the game was played
        season (str): season in which the game was played
        game_id (int): internal NHL game identification

    Returns: game_data (DataFrame): loaded json data

    """
    with open(os.path.join(f'{path}/{year}/{season}/', f'{game_id}.json'), 'r') as file:
        game_data = json.load(file)
    return game_data

def binary2cumulative(lst: list) -> list:
    """
    Arguments:
        lst (list)

    Returns: cumulative list wherein the element at index j is the sum of previous elements until reaching a previous 0 entry
            Ex: [0 1 1 1 1 0 1 1 1] -> [0 1 2 3 4 0 1 2 3]
    """
    nlst = []
    count = 0
    for i in lst:
        if i == 1:
            count += 1
        else:
            count = 0  
        nlst.append(count)
    return nlst

def power_play(df: pd.DataFrame, select_period: int) -> np.array:
    """
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
    """
        
    # loading Event indices for penalty plays
    idx = df['liveData']['plays']['penaltyPlays']

    # Determining away and home teams
    home_team = df['liveData']['boxscore']['teams']['home']['team']['triCode']
    away_team = df['liveData']['boxscore']['teams']['away']['team']['triCode']

    # Definiting total time (s) per standard period
    TotalT = 60*20
    
    # Initializing arrays to store time (s) since last penalty for all teams combined, away team and home team
    timer = np.zeros(TotalT)
    timer_away = np.zeros(TotalT)
    timer_home = np.zeros(TotalT)

    # Initializing arrays to store number of plays on the field for the home and away teams
    players_away = 5*np.ones(TotalT)
    players_home = 5*np.ones(TotalT)

    # Iterate through every penalty event and determining the time interval during which the penalty is in effect
    # Logging time since period start
    # Logging number of remaining players on the ice 
    time_elapsed = 0
    period_count = 0
    for i in idx:
        penalty_event = df['liveData']['plays']['allPlays'][i]
        period = penalty_event['about']['period']
        if select_period == period:
            # Computing the time remaining in the current period
            periodTimeRemaining =  60*int(penalty_event['about']['periodTimeRemaining'].split(':')[0]) + int(penalty_event['about']['periodTimeRemaining'].split(':')[1])
            # Penalty start time (s)
            penalty_t0 = 60*int(penalty_event['about']['periodTime'].split(':')[0]) + int(penalty_event['about']['periodTime'].split(':')[1])
            penalty_length = min(60*int(penalty_event['result']['penaltyMinutes']) , periodTimeRemaining)
            # Penalty end time (s)
            penalty_t1 = penalty_length + penalty_t0
            # Total length of the period / should correspond to TotalT
            periodLength_s = penalty_t0+periodTimeRemaining
            team = penalty_event['team']['triCode']

            # Storing data 
            if team == home_team:
                for i in range(time_elapsed+penalty_t0, time_elapsed+penalty_t1):
                    timer_home[i] = 1
                    timer[i] = 1
                    # Reduce by 1 given that player is in penalty box
                    players_home[i] = players_home[i]-1
            elif team == away_team:
                for i in range(time_elapsed+penalty_t0, time_elapsed+penalty_t1):
                    timer_away[i] = 1
                    timer[i] = 1
                    # Reduce by 1 given that player is in penalty box
                    players_away[i] = players_away[i]-1
      
    return binary2cumulative(timer), binary2cumulative(timer_home), binary2cumulative(timer_away), players_home, players_away


def is_goal(df: pd.DataFrame) -> bool:
    """
    Arguments:
        df (Dataframe): Database dataframe

    Returns: bool 0/1 corresponding to if the event was a goal (1) or a shot (0)
    """
    if df['eventType'] == 'Goal':
        return 1
    else:
        return 0

def feature_eng2_raw(DATA_PATH, year, season, game_id):
    """
    Arguments:
        path (str): DATA_PATH
        year (int): year in which the game was played
        season (str): season in which the game was played
        game_id (int): internal NHL game identification

    Returns: df_eng2 (DataFrame) containing play-by-play data as well as previous play data, as required
                by Milestone 2 part 4, subsection 1,2,3 AS WELL AS BONUS

    """
    df = tidy_data(DATA_PATH, year, season, game_id)
    if len(df) == 0:
        return
    df_eng2 = df

    # Converting periodTime (mm:ss) to gameSeconds (s), i.e., time (s) since game start!
    df_eng2[['minutes', 'seconds']] = df_eng2['periodTime'].str.split(':', expand=True)
    df_eng2['gameSeconds'] = df['period']*(df_eng2['minutes'].astype(int) * 60 + df_eng2['seconds'].astype(int))

    df_eng2 = df_eng2.drop(columns=['minutes', 'seconds'])

    # Copying df and shifting by one below to get previous play data
    df_copy = df
    new_df = df_copy.shift(fill_value=np.nan)
    new_df.columns = df.columns

    new_df.iloc[0, :] = np.nan
    new_df.iloc[:, 0] = np.nan

    # Keeping pertinent previous play data
    df_copy = new_df
    df_eng2['LastEventType'] = df_copy['eventType']
    df_eng2['Last_x_coordinate'] = df_copy['x_coordinate']
    df_eng2['Last_y_coordinate'] = df_copy['y_coordinate']
    df_eng2['Last_gameSeconds'] = df_copy['gameSeconds']
    df_eng2['Last_period'] = df_copy['period']
    # Computing distance between last event and current event
    df_eng2['DistanceLastEvent'] = np.sqrt((df_eng2['Last_x_coordinate']-df_eng2['x_coordinate'])**2+(df_eng2['Last_y_coordinate']-df_eng2['y_coordinate'])**2)
    df_eng2['Rebound'] = df_eng2['LastEventType'] == 'Shot'
    df_eng2['LastShotAngle'] = df_copy['shotAngle']
    # Computing the change in shot angle between last and current event
    df_eng2['changeShotAngle'] = df_eng2['LastShotAngle']+df_eng2['shotAngle']
    df_eng2['timeFromLastEvent'] = df_eng2['gameSeconds']-df_eng2['Last_gameSeconds']
    # Computing speed at distance from last event divided by time since last event
    df_eng2['speed'] = df_eng2['DistanceLastEvent']/df_eng2['timeFromLastEvent']
    
    # Keeping only values from powerplay function which correspond to a time at which an event occured
    dfg = game_data(DATA_PATH, year, season, game_id)
    period_lst = df_eng2['period'].tolist()

    # Initializing arrays for bonus features occuring at times at which events also occured
    time_since_pp = np.array([[]])
    no_players_home = np.array([[]])
    no_players_away = np.array([[]])

    # Looping through all periods
    for select_period in range(min(period_lst),max(period_lst)+1):
        # Get df indices for events occuring in select_period
        selected_idx = np.where(np.array(period_lst) == select_period)
        selected_times = df_eng2['gameSeconds'].iloc[selected_idx]
        
        
        # Determine period time at which events occured
        selected_times = np.array(selected_times%(60*20))

        # Get bonus features at selected_times
        time_since_pp = np.concatenate((time_since_pp, np.array(power_play(dfg, select_period)[0])[selected_times]), axis=None)
        no_players_home = np.concatenate((no_players_home, np.array(power_play(dfg, select_period)[3])[selected_times]), axis=None)
        no_players_away = np.concatenate((no_players_away, np.array(power_play(dfg, select_period)[4])[selected_times]), axis=None)

    # Store bonus features at event times
    df_eng2['time_since_pp'] = time_since_pp
    df_eng2['no_players_home'] = no_players_home
    df_eng2['no_players_away'] = no_players_away
    df_eng2['is_goal'] = df_eng2.apply(is_goal, axis=1)
    

    # Keeping track of number of goals scored by home and away teams
    df_eng2['home_pts'] = 0
    df_eng2['away_pts'] = 0

    # Grouping by game_id to be sure to keep track of points within a given game
    # Summing goals scored by the home and away teams, respectively
    home_pts_cumsum = df_eng2[df_eng2['team'] == df_eng2['homeTeam']].groupby(['game_id', 'homeTeam'])['is_goal'].cumsum()
    away_pts_cumsum = df_eng2[df_eng2['team'] == df_eng2['awayTeam']].groupby(['game_id', 'awayTeam'])['is_goal'].cumsum()

    df_eng2 = pd.merge(df_eng2, home_pts_cumsum, left_index=True, right_index=True, suffixes=('', '_home_cumsum'), how='left')
    df_eng2 = pd.merge(df_eng2, away_pts_cumsum, left_index=True, right_index=True, suffixes=('', '_away_cumsum'), how='left')
    df_eng2['home_pts'] = df_eng2.groupby(['game_id', 'homeTeam'])['is_goal_home_cumsum'].ffill().fillna(0).astype(int)
    df_eng2['away_pts'] = df_eng2.groupby(['game_id', 'awayTeam'])['is_goal_away_cumsum'].ffill().fillna(0).astype(int)
    df_eng2.drop(['is_goal_home_cumsum', 'is_goal_away_cumsum'], axis=1, inplace=True)

    # Computing goal differentials
    df_eng2['diff_pts'] = df_eng2['home_pts'] - df_eng2['away_pts']

    # Keeping 'is_goal' as last (target) column
    df_eng2['is_goal'] = df_eng2.pop('is_goal')

    return df_eng2

# Get json files in parent directory
json_regex = re.compile(r"(.*)\.json")

def season_integration_eng2(path: str, year: int, season: str) -> pd.DataFrame:
    """
    Arguments:
        path (str): DATA_PATH
        year (int): year in which the game was played
        season (str): season in which the game was played

    Returns: df (DataFrame) containing data from all games in chosen season

    """
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


def feature_eng2(path, year) -> pd.DataFrame:
    """
    Arguments:
        path (str): DATA_PATH
        year (int): year in which the game was played

    Returns: df (Dataframe): df corresponding to feature_eng2_raw features for all games in selected year

    """
    df = pd.DataFrame()
    for season in ['regular', 'playoffs']:
        season_df = season_integration_eng2(path, year, season)
        df = pd.concat([df, season_df], ignore_index=True)
    return df

def feature_eng2_cleaned(path, year) -> pd.DataFrame:
    """
    Arguments:
        path (str): DATA_PATH
        year (int): year in which the game was played

    Returns: df (Dataframe): df corresponding to feature_eng2_raw features for all games in selected year,
                                but for which we keep only relevant (non-categorical) features

    """
    df = pd.DataFrame()
    for season in ['regular', 'playoffs']:
        season_df = season_integration_eng2(path, year, season)
        df = pd.concat([df, season_df], ignore_index=True)
    df['is_goal'] = df.pop('is_goal')
    return df[['gameSeconds','period','x_coordinate','y_coordinate','shotDistance','shotAngle','shotType','LastEventType','Last_x_coordinate','Last_y_coordinate','timeFromLastEvent','DistanceLastEvent','Rebound','changeShotAngle','speed','time_since_pp','no_players_home','no_players_away', 'home_pts', 'away_pts', 'diff_pts', 'is_goal']]

def get_train_data(DATA_PATH: str, return_optional_features):
    """
    Arguments:
        DATA_PATH (str): path to data

    Returns: saves csv of data (from feature_eng2_cleaned) from 2016 to 2019 for training

    """
    data = feature_eng2_cleaned(DATA_PATH, 2016)
    data = pd.concat([data, feature_eng2_cleaned(DATA_PATH, 2017)], ignore_index=True)
    data = pd.concat([data, feature_eng2_cleaned(DATA_PATH, 2018)], ignore_index=True)
    data = pd.concat([data, feature_eng2_cleaned(DATA_PATH, 2019)], ignore_index=True)

    if return_optional_features:
        data.to_csv(DATA_PATH + '/clean_train_data_with_optional.csv')
    else:
        data = data.drop(['away_pts', 'diff_pts', 'home_pts'], axis=1)
        data.to_csv(DATA_PATH + '/clean_train_data.csv')

def get_test_data(DATA_PATH, season, return_optional_features):
    """
        Arguments:
            DATA_PATH (str): path to data

        Returns: saves csv of data (from feature_eng2_cleaned) from 2020 for testing

    """
    data = season_integration_eng2(DATA_PATH, 2020, season)
    data = data[['gameSeconds', 'period', 'x_coordinate', 'y_coordinate', 'shotDistance', 'shotAngle', 'shotType',
                 'LastEventType', 'Last_x_coordinate', 'Last_y_coordinate', 'timeFromLastEvent', 'DistanceLastEvent',
                 'Rebound', 'changeShotAngle', 'speed', 'time_since_pp', 'no_players_home', 'no_players_away',
                 'home_pts', 'away_pts', 'diff_pts', 'is_goal']]
    if return_optional_features:
        data.to_csv(DATA_PATH + f'/clean_test_data_{season}_with_optional.csv')
    else:
        data = data.drop(['away_pts', 'diff_pts', 'home_pts'], axis=1)
        data.to_csv(DATA_PATH + f'/clean_test_data_{season}.csv')

def get_full_test_data(DATA_PATH, season, return_optional_features):
    """
        Arguments:
            DATA_PATH (str): path to data

        Returns: saves csv of data (from feature_eng2) from 2020 for testing

    """
    data = season_integration_eng2(DATA_PATH, 2020, season)

    if return_optional_features:
        data.to_csv(DATA_PATH + f'/full_test_data_{season}_with_optional.csv')
    else:
        data = data.drop(['away_pts', 'diff_pts', 'home_pts'], axis=1)
        data.to_csv(DATA_PATH + f'/full_test_data_{season}.csv')

def get_full_train_data(DATA_PATH: str, return_optional_features):
    """
    Arguments:
        DATA_PATH (str): path to data

    Returns: saves csv of data (from feature_eng2_cleaned) from 2016 to 2019 for training

    """
    data = feature_eng2(DATA_PATH, 2016)
    data = pd.concat([data, feature_eng2(DATA_PATH, 2017)], ignore_index=True)
    data = pd.concat([data, feature_eng2(DATA_PATH, 2018)], ignore_index=True)
    data = pd.concat([data, feature_eng2(DATA_PATH, 2019)], ignore_index=True)

    if return_optional_features:
        data.to_csv(DATA_PATH + '/full_train_data_with_optional.csv')
    else:
        data = data.drop(['away_pts', 'diff_pts', 'home_pts'], axis=1)
        data.to_csv(DATA_PATH + '/full_train_data.csv')

if __name__ == "__main__":
    PATH = '../../IFT6758_Data'

    dataset = input('Enter train for parsing train data and test for parsing test data: ').strip()
    assert dataset in ['train', 'test']

    subset_or_full = input('Enter subset for only feature engineering 2 data and full for all data: ').strip()
    assert subset_or_full in ['subset', 'full']

    optional_features = input('Enter yes for optional features. Otherwise enter no: ').strip()
    assert optional_features in ['yes', 'no']

    if dataset == 'test':
        season = input('Enter season as regular or playoffs: ').strip()
        assert season in ['regular', 'playoffs']

    if dataset == 'train':
        if subset_or_full == 'subset':
            get_train_data(PATH, optional_features == 'yes')
        else:
            get_full_train_data(PATH, optional_features == 'yes')
    else:
        if subset_or_full == 'subset':
            get_test_data(PATH, season, optional_features == 'yes')
        else:
            get_full_test_data(PATH, season, optional_features == 'yes')