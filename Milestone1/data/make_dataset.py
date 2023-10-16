import os
import json

import requests as req
import time
from enum import Enum


class Gametype(Enum):
    REGULAR = '02'
    PLAYOFFS = '03'
#     PRESEASON = '01'
#     ALLSTAR = '04'

class scrape_nhl_data:
    def get_game_id(self, season: str, game_type: str, game_number: str):
        return f'{season}{game_type}{str(game_number).zfill(4)}'

    def write_data(self, loc: str, season: str, content):
        if season != Gametype.REGULAR.name and 'endDateTime' not in content['gameData']['datetime']:
            return
        with open(f'{loc}.json', 'w+', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)

    def scrape_data(self, game_id, game_type, path):
        endpoint = f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/'
        try:
            time.sleep(0.5)
            res = req.get(endpoint)
            res.raise_for_status()
            self.write_data(f'{path}/{game_id}', game_type, res.json())
        except req.exceptions.HTTPError as err:
            print(f'API failed for {game_id} with status code {err.response.status_code}')
        except Exception as e:
            print(f'{game_type} trace: {endpoint} {game_id}')
            print(e)

    def get_play_by_play_data(self,
                              path: str,
                              seasons_to_game_volume_map: dict,
                              game_types: list
                              ):
        """Creates folders for each game type in a season and stores the play-by-play data for
        each game in a json file
                Arguments:
                    path (str): Location where the files should be created.
                    Ideally it should be the 'data' folder of our repository.

                    Note: Do not precede the path with a '/'. If the data
                    needs to be saved in the same directory as this script then
                    pass an empty string ''.

                    seasons_to_game_volume_map (dict of str: int): Map of seasons
                    for which the data is required and the corresponding number of
                    games in that season. For e.g. it will have the key as '2016'
                    for the 2016-17 season and 1230 as the corresponding number of games.

                    game_types (dict of str: str): List of game types for which
                    data needs to be retrieved.

                Return:
                    Folder containing data for each hockey season. These folders in
                    turn contain play-by-play data for regular and playoff games.
        """
        # Loop in a single hockey season e.g. 2016 (2016-17 season), 2020 (2020-21 season)
        for season, games in seasons_to_game_volume_map.items():
            # Loop inside a particular game type i.e. regular or playoffs
            for game_type in game_types:
                if len(path.strip()) == 0:
                    loc = f'IFT6758_Data/{season}/{game_type}'
                else:
                    loc = f'{path}/IFT6758_Data/{season}/{game_type}'

                if not os.path.exists(loc):
                    os.makedirs(loc)

                # Check the game type. If it is regular then the last 4 digits
                # should be the game number
                if game_type == Gametype.REGULAR.name:
                    for game_number in range(1, games + 1):
                        game_id = self.get_game_id(season, Gametype.REGULAR.value, str(game_number))
                        self.scrape_data(game_id, Gametype.REGULAR.name, loc)

                # Otherwise, game_type == 'playoff' and the last 4 digits
                # should be composed as follows:
                # first 2 digits -> round number (can be 01, 02, 03, 04)
                # third digit -> match up (can be upto 8, 4, 2, 1 for
                # the above mentioned round numbers)
                # fourth digit -> game number (can be from 1 to 7)
                else:
                    total_match_ups = 8
                    round_num = 1

                    # Continually divide total_match_ups as after each round
                    # half of the teams are eliminated
                    while total_match_ups != 0:
                        for match_up in range(1, total_match_ups + 1):
                            for game_number in range(1, 8):
                                game_id = self.get_game_id(season, Gametype.PLAYOFFS.value,
                                                           f'{str(round_num).zfill(2)}{match_up}{game_number}')
                                self.scrape_data(game_id, Gametype.PLAYOFFS.name, loc)
                        total_match_ups = total_match_ups // 2
                        round_num += 1

if __name__ == "__main__":
    scraper = scrape_nhl_data()
    season_data = {'2016': 1230, '2017': 1271, '2018': 1271, '2019': 1271, '2020': 868}
    game_types = [Gametype.REGULAR.name, Gametype.PLAYOFFS.name]
    scraper.get_play_by_play_data(path='', seasons_to_game_volume_map=season_data, game_types=game_types)