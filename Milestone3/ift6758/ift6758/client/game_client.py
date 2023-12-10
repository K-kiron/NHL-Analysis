import json
import requests
import pandas as pd
import logging
import numpy as np


logger = logging.getLogger(__name__)
headers = {'Content-Type': 'application/json'}

class GameClient:
    def __init__(self):
        self.base_url = f"http://localhost:5000"
        logger.info(f"Initializing client; base URL: {self.base_url}")

    def ping_game(self, game_id: int) -> pd.DataFrame:
        """
        Query the NHL API for live game data and process new events.

        Args:
            game_id (int): Input game_id to get the game data.
        """
        # Fetching live game data from the NHL API
        try:
            response = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
            response.raise_for_status()
            game_data = response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching data for game ID {game_id}: {e}")
            return pd.DataFrame()

        preprocessed_data = preprocessing(game_data)

        processed_new_events = process_events(preprocessed_data)

        return processed_new_events
    

def preprocessing(game_data: json) -> pd.DataFrame:
    """
    Preprocess the new events and return the preprocessed data.

    Args:
        game_data (json): The game data.

    Return:
        preprocessed_new_events (pd.DataFrame): The preprocessed new events.
    """

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


def process_events(preprocessed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the new events and return the processed data.

    Args:
        preprocessed_data (pd.DataFrame): The preprocessed data.

    Return:
        processed_new_events (pd.DataFrame): The processed new events.
    """
    pass
    

