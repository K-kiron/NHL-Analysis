import json
import os
import glob
import pandas as pd
import numpy as np
import math

def tidy_data(path, year, season, game_id):
    with open(os.path.join(f'{path}/{year}/{season}/', f'{game_id}.json'), 'r') as file:
        game_data = json.load(file)
    
    shot_data_temp = []
    
    playData = game_data['liveData']['plays']['allPlays']
    homeTeam = game_data['gameData']['teams']['home']['triCode']
    awayTeam = game_data['gameData']['teams']['away']['triCode']
    for i in range(len(playData)):
        eventData = playData[i]
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
                if period > 3:
                    period = 4
                try:
                    goalLocation = game_data['liveData']['linescore']['periods'][period-1]['home']['rinkSide']
                except KeyError:
                    goalLocation = None
                    
            elif team == awayTeam:
                if period > 3:
                    period = 4
                try:
                    goalLocation = game_data['liveData']['linescore']['periods'][period-1]['away']['rinkSide']
                except KeyError:
                    goalLocation = None
                    
            if goalLocation == 'left' and y_coordinate is not None:
                shotangle = np.degrees(np.arctan2(np.abs(y_coordinate), np.abs(x_coordinate - 89)))
            elif goalLocation == 'right' and y_coordinate is not None:
                shotangle = np.degrees(np.arctan2(np.abs(y_coordinate), np.abs(x_coordinate + 89)))
            elif goalLocation == None:
                shotangle = None
                
            if goalLocation == 'left' and y_coordinate is not None:
                shotDistance = np.sqrt(y_coordinate**2 + (x_coordinate - 89)**2)
            elif goalLocation == 'right' and y_coordinate is not None:
                shotDistance = np.sqrt(y_coordinate**2 + (x_coordinate + 89)**2)
            elif goalLocation == None:
                shotDistance = None
            
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

            # if shotDistance is not None and shotangle is not None:
            #     if shotDistance > 100:
            #         print(shotDistance)

            shot_data_temp.append(shot_data)
            
    return pd.DataFrame(shot_data_temp)#.dropna()