import json
import os
import glob
import pandas as pd

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
                if period%2 != 0:
                    goalLocation = 'Left'
                else:
                    goalLocation = 'Right'
                    
            elif team == awayTeam:
                if period%2 != 0:
                    goalLocation = 'Right'
                else:
                    goalLocation = 'Left'

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
                'strength': strength
            }

            shot_data_temp.append(shot_data)
            
    return pd.DataFrame(shot_data_temp)

#test case below
#print(tidy_data(2017, 'regular', 2017020001))
