import json
import os
import glob
import pandas as pd

def tidy_data(path, year, season, game_id):
    with open(os.path.join(f'{path}/{year}/{season}/', f'{game_id}.json'), 'r') as file:
        game_data = json.load(file)
    
    shot_data_temp = []
    
    playData = game_data['liveData']['plays']['allPlays']
    for i in range(len(playData)):
        eventData = playData[i]
        if eventData['result']['event'] == 'Shot' or eventData['result']['event'] == 'Goal':
            game_id = game_data['gameData']['game']['pk']
            periodType = eventData['about']['periodType']
            period = eventData['about']['period']
            periodTime = eventData['about']['periodTime']
            team = eventData['team']['name']
            eventType = eventData['result']['event']
            coordinates = eventData['coordinates']
            shooter = eventData['players'][0]['player']['fullName']
            goalie = eventData['players'][-1]['player']['fullName']
            shotType = eventData['result']['secondaryType']
            if eventType == 'Goal':
                emptyNet = eventData['result']['emptyNet']
                strength = eventData['result']['strength']['code']
            else:
                emptyNet = None
                strength = None

            # Store the data in a dictionary
            shot_data = {
                'game_id': game_id,
                'periodType': periodType,
                'period': period,
                'periodTime': periodTime,
                'team': team,
                'eventType': eventType,
                'coordinates': coordinates,
                'shooter': shooter,
                'goalie': goalie,
                'shotType': shotType,
                'emptyNet': emptyNet,
                'strength': strength
            }

            shot_data_temp.append(shot_data)
            
    return pd.DataFrame(shot_data_temp)

#test case below
#print(tidy_data([path], 2017, 'regular', 2017020001))
