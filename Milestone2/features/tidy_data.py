import json
import os
import glob
import pandas as pd
import numpy as np
import math

def tidy_data(path: str, year: int, season: str, game_id: int) -> pd.DataFrame:
    '''
    Arguments:
        path (str): DATA_PATH
        year (int): year in which the game was played
        season (str): season in which the game was played
        game_id (int): internal NHL game identification

    Returns: DataFrame containing game data for all events
                All features are listed below under shot_data
        
    '''

    # Loading json file
    with open(os.path.join(f'{path}/{year}/{season}/', f'{game_id}.json'), 'r') as file:
        game_data = json.load(file)
    
    shot_data_temp = []
    
    # Loading play Data
    playData = game_data['liveData']['plays']['allPlays']
    homeTeam = game_data['gameData']['teams']['home']['triCode']
    awayTeam = game_data['gameData']['teams']['away']['triCode']

    # Looping through events
    for i in range(len(playData)):
        eventData = playData[i]
        # Get data for shot/goal events
        if eventData['result']['event'] == 'Shot' or eventData['result']['event'] == 'Goal':
            game_id = game_data['gameData']['game']['pk']
            periodType = eventData['about']['periodType']
            period = eventData['about']['period']
            periodTime = eventData['about']['periodTime']
            team = eventData['team']['triCode']
            eventType = eventData['result']['event']
            try:
                x_coordinate = eventData['coordinates']['x']
                y_coordinate = eventData['coordinates']['y']
            except KeyError:
                x_coordinate = None
                y_coordinate = None
            shooter = eventData['players'][0]['player']['fullName']
            goalie = eventData['players'][-1]['player']['fullName']
            try:
                shotType = eventData['result']['secondaryType']
            except KeyError:
                shotType = None
            if eventType == 'Goal':
                try:
                    emptyNet = eventData['result']['emptyNet']
                except KeyError:
                    emptyNet = None
                strength = eventData['result']['strength']['code']
            else:
                emptyNet = None
                strength = None
                
            if team == homeTeam:
                if period%2 != 0:
                    goalLocation = 'Left'
                else:
                    goalLocation = 'Right'
                    
            elif team == awayTeam:
                if period%2 != 0:
                    goalLocation = 'Right'
                else:
                    goalLocation = 'Left'
            
            # Computing shot angle based on goal location
            if goalLocation == 'Left' and y_coordinate is not None:
                shotangle = np.degrees(np.arctan2(np.abs(y_coordinate), np.abs(x_coordinate + 89)))
            elif goalLocation == 'Right' and y_coordinate is not None:
                shotangle = np.degrees(np.arctan2(np.abs(y_coordinate), np.abs(x_coordinate - 89)))
                
            if goalLocation == 'Left' and y_coordinate is not None:
                shotDistance = np.sqrt(y_coordinate**2 + (x_coordinate + 89)**2)
            elif goalLocation == 'Right' and y_coordinate is not None:
                shotDistance = np.sqrt(y_coordinate**2 + (x_coordinate - 89)**2)
            
            # Storing features
            shot_data = {
                'game_id': game_id,
                'homeTeam': homeTeam,
                'awayTeam': awayTeam,
                'periodType': periodType,
                'period': period,
                'periodTime': periodTime,
                'team': team,
                'eventType': eventType,
                'x_coordinate': x_coordinate,
                'y_coordinate': y_coordinate,
                'goal_location': goalLocation,
                'shooter': shooter,
                'goalie': goalie,
                'shotType': shotType,
                'emptyNet': emptyNet,
                'strength': strength,
                'shotAngle': shotangle,
                'shotDistance': shotDistance
            }

            shot_data_temp.append(shot_data)
            
    return pd.DataFrame(shot_data_temp)