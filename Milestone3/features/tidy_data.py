import json
import os
import glob
import pandas as pd
import numpy as np
import math

variable_translation = {
    'shot-on-goal': 'Shot',
    'goal': 'Goal',
    'snap': 'Snap Shot',
    'wrist': 'Wrist Shot',
    'backhand': 'Backhand',
    'tip-in': 'Tip-In',
    'slap': 'Slap Shot',
    'REG': 'REGULAR',
    'OT': 'OVERTIME',
}


def compute_goal_data(goalLocation: str, x_coordinate: float, y_coordinate: float):
    """
    Fctn computing shot angle and shot distance
    Arguments:
        goalLocation (str): goal location ('left' or 'right')
        x_coordinate (float): x coordinate of shooter 
        y_coordinate (float): y coordinate of shooter 

    Returns: shotDistance (float), shotAngle (float)

    """

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

    return shotDistance, shotangle

def tidy_data(path, year, season, game_id):
    """
    Fctn computing all event features for a given game
    These are game_id homeTeam awayTeam periodType  period periodTime team eventType \ \
    x_coordinate  y_coordinate goal_location  shooter  \
    goalie    shotType emptyNet strength  shotAngle  shotDistance 
    Arguments:
        path (str): path to data
        year (int): game year
        season (str): "regular" or "playoff"
        game id (int): game ID number from NHL API assignment

    Returns: pd.Dataframe
    """

    with open(os.path.join(f'{path}/{year}/{season}/', f'{game_id}.json'), 'r') as file:
        game_data = json.load(file)
    
    shot_data_temp = []
    
    playData = game_data['plays']
    #homeTeam = game_data['gameData']['teams']['home']['triCode']
    homeTeam = game_data['homeTeam']['abbrev']
    awayTeam = game_data['awayTeam']['abbrev']
    for i in range(len(playData)):
        eventData = playData[i]
        period_count = 1
        #if eventData['result']['event'] == 'Shot' or eventData['result']['event'] == 'Goal':
        if eventData['typeDescKey'] == 'shot-on-goal' or eventData['typeDescKey'] == 'goal':
            #game_id = game_data['gameData']['game']['pk']
            game_id = game_data['id']
            #periodType = eventData['about']['periodType']
            periodType = variable_translation.get(eventData['periodDescriptor']['periodType'],eventData['periodDescriptor']['periodType'])
            #period = eventData['about']['period']
            period = eventData['periodDescriptor']['number']
            #periodTime = eventData['about']['periodTime']
            periodTime = eventData['timeInPeriod']
            #team = eventData['team']['triCode']
            team_id = eventData['details'].get('eventOwnerTeamId', None)
            if team_id == game_data['homeTeam']['id']:
                team = game_data['homeTeam']['abbrev']
            elif team_id == game_data['awayTeam']['id']:
                team = game_data['awayTeam']['abbrev']
            else:
                team = None
            #eventType = eventData['result']['event']
            eventType = variable_translation.get(eventData['typeDescKey'], eventData['typeDescKey'])
            try:
                #x_coordinate = eventData['coordinates']['x']
                x_coordinate = eventData['details']['xCoord']+0.0
                #y_coordinate = eventData['coordinates']['y']
                y_coordinate = eventData['details']['yCoord']+0.0
            except KeyError:
                x_coordinate = None
                y_coordinate = None

            goalie_id = eventData['details'].get('goalieInNetId', None)
            goalie_data = next((player for player in game_data['rosterSpots'] if player['playerId'] == goalie_id), None)
            if goalie_id is not None:
                goalie = goalie_data['firstName']['default'] + ' ' + goalie_data['lastName']['default']
            else:
                goalie = None

            if eventData['typeDescKey'] == 'shot-on-goal':
                shooter_id = eventData['details'].get('shootingPlayerId', None)
            elif eventData['typeDescKey'] == 'goal':
                shooter_id = eventData['details'].get('scoringPlayerId', None)

            shooter_data = next((player for player in game_data['rosterSpots'] if player['playerId'] == shooter_id), None)
            if shooter_id is not None:
                shooter = shooter_data['firstName']['default'] + ' ' + shooter_data['lastName']['default']
            else:
                shooter = None

            #shooter = eventData['players'][0]['player']['fullName']
            #goalie = eventData['players'][-1]['player']['fullName']
            try:
                #shotType = eventData['result']['secondaryType']
                shotType = variable_translation.get(eventData['details']['shotType'], eventData['details']['shotType'])
            except KeyError:
                shotType = None

            strength = (eventData['situationCode'][1],eventData['situationCode'][2])
            if eventType == 'Goal':
                try:
                    #emptyNet = eventData['result']['emptyNet']
                    emptyNet = 'goalieInNetId' not in eventData['details']
                except KeyError:
                    emptyNet = False
            #    strength = eventData['result']['strength']['code']
            else:
                emptyNet = False
            #    strength = None
            
            if eventData['details']["zoneCode"] == 'D' or eventData['details']["zoneCode"] == 'O':
                if eventData['details']["zoneCode"] == 'D':
                    if x_coordinate < 0:
                        goal_location = 'left'
                    else:
                        goal_location = 'right'
                elif eventData['details']["zoneCode"] == 'O':
                    if x_coordinate < 0:
                        goal_location = 'right'
                    else:
                        goal_location = 'left'

            else:
                goal_location = None
                
            shotDistance, shotAngle = compute_goal_data(goal_location, x_coordinate, y_coordinate)
            
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
                'goal_location': goal_location,
                'shooter': shooter,
                'goalie': goalie,
                'shotType': shotType,
                'emptyNet': emptyNet,
                'strength': strength,
                'shotAngle': shotAngle,
                'shotDistance': shotDistance
            }

            if shotDistance is not None and shotAngle is not None:
                if shotDistance > 100:
                    print(shotDistance)

            shot_data_temp.append(shot_data)

    # Fixing NONE vals in goal location
    df = pd.DataFrame(shot_data_temp)
    for idx, row in df.iterrows():
        if pd.isnull(row['goal_location']):
            team = row['team']
            period = row['period']
            
            sample_goal = df[(df['team'] == team) & (df['period'] == period) & (~df['goal_location'].isnull())]
            if not sample_goal.empty:
                goal_location = sample_goal.iloc[0]['goal_location']
                shotDistance, shotAngle = compute_goal_data(goal_location, x_coordinate, y_coordinate)
                
                df.at[idx, 'goal_location'] = goal_location
                df.at[idx, 'shotDistance'] = shotDistance
                df.at[idx, 'shotAngle'] = shotAngle
            
    return df#.dropna()
