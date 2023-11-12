import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


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
        binary2cumulative(timer_home): array where the entry is the time since the power play started for penalties
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
    players_away = 6*np.ones(TotalT)

    timer_home = np.zeros(TotalT)
    players_home = 6*np.ones(TotalT)

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
