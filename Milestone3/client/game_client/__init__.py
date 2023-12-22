import json
import requests
import pandas as pd
# import logging
import numpy as np
import sys
import os

import sys
sys.path.append('../features')
from tidy_data import compute_goal_data

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

headers = {'Content-Type': 'application/json'}

def is_identical(file1_path: str, file2: json) -> bool:
    """
    Check if two files are identical.

    Args:
        file1_path (str): Path to the old file.
        file2 (json): The new file.

    Returns:
        bool: True if the files are identical, False otherwise.
    """

    with open(file1_path, 'r') as file1:
        file1_data = json.load(file1)
        return file1_data == file2
            
def read_update_partial_data(update_file_path: str, old_file_path: str) -> json:
    """
    Read the update file and return the updated data.

    Args:
        update_file_path (str): Path to the update file.
        old_file_path (str): Path to the old file.

    Returns:
        json: The updated data.
    """
    with open(update_file_path, 'r') as update_file:
        update_data = json.load(update_file)
    with open(old_file_path, 'r') as old_file:
        old_data = json.load(old_file)

    updated_data = {key: update_data[key] for key in update_data if key not in old_data or update_data[key] != old_data[key]}
    return updated_data

class GameClient:
    def __init__(self):
        self.base_url = f"http://serving:8000"

    def ping_game(self, game_id) -> pd.DataFrame:
        """
        Query the NHL API for live game data and process events.

        Returns:
            processed new events (pd.DataFrame): The processed new events, if any.
        """
        # Fetching live game data from the NHL API
        try:
            response = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
            response.raise_for_status()
            game_data = response.json()

            if not os.path.exists('live_raw'):
                os.makedirs('live_raw')

            file = f'live_raw/{game_id}.json'
            
            if not os.path.exists(file):
                print("Loading new game data...")
                with open(file, 'w') as f:
                    json.dump(game_data, f)
                preprocessed_data = preprocessing(game_data)
            else:
                if not is_identical(file, game_data):
                    print("There is an update in the game data!")
                    with open(file, 'w') as f:
                        json.dump(game_data, f)
                    game_data = read_update_partial_data(game_data, file)
                    preprocessed_data = preprocessing(game_data)
                else:
                    print("No update in the game data!")
                    preprocessed_data = preprocessing(game_data)

            return {
                'home_team_name': game_data['homeTeam']['name']['default'],
                'away_team_name': game_data['awayTeam']['name']['default'],
                'home_team_code': game_data['homeTeam']['abbrev'],
                'away_team_code': game_data['awayTeam']['abbrev'],
                'home_team_score': game_data['homeTeam']['score'],
                'away_team_score': game_data['awayTeam']['score'],
                'data': preprocessed_data,
                "status_code": 200
            }

        except requests.RequestException as e:
            return {'status_code': 400}

        
    

def preprocessing(game_data: json) -> pd.DataFrame:
    """
    Preprocess the new events and return the preprocessed data.

    Args:
        game_data (json): The game data.

    Return:
        preprocessed_new_events (pd.DataFrame): The preprocessed new events.
    """

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

    

if __name__ == "__main__":
    client = GameClient(2022030411)
    client.ping_game()

