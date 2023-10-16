import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re
from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import plotly.express as px
#from dash import Dash, dcc, html
sys.path.append("IFT6758B-Project-B10/Milestone1/features")

from tidy_data import tidy_data
def load_tidy_data(year, season):
    path = f"IFT6758B-Project-B10/Milestone1/data/IFT6758_Data"

    # Define game_type based on season
    game_type = '02' if season.lower() == 'regular' else '03'

    all_data = []

    # List all files in the directory
    files_in_directory = os.listdir(os.path.join(path, year, season))

    for file_name in files_in_directory:
        if file_name.endswith(".json"):
            # Extract the game_id from the file name (assuming the file names follow the pattern)
            game_id = int(file_name.split(".")[0])

            # Load the data and append it to the list
            game_data = tidy_data(path, year, season, game_id)
            if game_data is not None:
                all_data.append(game_data)

    # Concatenate all the data in the list
    if all_data:
        all_data = pd.concat(all_data)
    else:
        # Handle the case where no data was loaded
        print(f"No data found for year {year} and game type {game_type}")
        return None

    return all_data

# Function to load the rink image
def load_rink_image(image_path):
    return Image.open("IFT6758B-Project-B10/Milestone1/visualizations/nhl_rink.png")

# Define offensive zone coordinates (adjust these based on your specific rink dimensions)
offensive_zone_x_min = 0
offensive_zone_x_max = 100
offensive_zone_y_min = -42.5
offensive_zone_y_max = 42.5



# Function to create a shot map on the rink image for a selected team
def create_shot_map_for_team(data, rink_image, selected_team):
    fig = go.Figure()


    # Define scale factors based on the rink image dimensions and scaling
    scale_factor_x = 1 #38.81 / 6100  # Assuming rink width is 200 units
    scale_factor_y = 1 #16.47 / 3000  # Assuming rink height is 85 units

    team_data = data[data['team'] == selected_team]
    X = []
    Y = []
    X_orig = []
    Y_orig = []
    for _, shot in team_data.iterrows():
        x = shot['x_coordinate']
        y = shot['y_coordinate']

        X_orig.append(x)
        Y_orig.append(y)
        if shot['goal_location'] == "Left":
            x = -x
            y = -y
        color = 'red' if shot['eventType'] == 'Goal' else 'blue'
        #print (x,y)
        # Check if the shot is in the offensive zone
        if (offensive_zone_x_min <= x <= offensive_zone_x_max) and (offensive_zone_y_min <= y <= offensive_zone_y_max):
            # Calculate adjusted coordinates with offsets
            x_adjusted = x * scale_factor_x
            y_adjusted = y * scale_factor_y

            X.append(x_adjusted)
            Y.append(y_adjusted)

        # Add a scatter point to the figure
       #fig.add_trace(go.Scatter(x=[x_adjusted], y=[y_adjusted], mode='markers', marker=dict(color=color, size=10)))
    print("max in X {}".format(max(X_orig)))
    print("max in Y {}".format(max(Y_orig)))

    print("min in X {}".format(min(X_orig)))
    print("min in Y {}".format(min(Y_orig)))

    sns.set_style("white")
    #sns.cubehelix_palette(as_cmap=True)
    sns.kdeplot(x=X, y=Y,cmap='icefire', shade=True, bw_adjust=.5,cbar=True,alpha=0.6)
    plt.ylim(-42.5,42.5)
    plt.xlim(-100,100)
    plt.imshow(rink_image, extent=[-100,100,-42.5, 42.5], aspect='auto')

    plt.show()
    #plt.scatter(x=X,y=Y)

    # Add surface trace
    #fig.add_trace(go.Surface(z=df.values.tolist(), colorscale="Viridis"))
    # Customize the figure layout
    fig.update_layout(
        width=800,
        height=900,
        autosize=False,
        margin=dict(t=0, b=0, l=0, r=0),
        template="plotly_white",
        title=f"Shot Map for {selected_team}",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
    )

    # Display the rink image as a background image
    fig.add_layout_image(
        source=rink_image,
        xref="x",
        yref="y",
        x=0,
        y=85,  # Adjust this value based on the image positioning
        sizex=200,
        sizey=85,
        opacity=1,
        layer="below"
    )

    return fig


if __name__ == "__main__":
    year = input("Enter a year: ")  # Prompt the user for the year
    season = input("Enter a season (e.g., 'REGULAR' or 'PLAYOFFS'): ")  # Prompt the user for the season input

    # Map season input to corresponding season code
    game_type = {'REGULAR': '02', 'PLAYOFFS': '03'}
    game_type = game_type.get(season.upper())  # Convert to uppercase for case-insensitive matching

    if season is None:
        print("Invalid season input. Please enter 'REGULAR' or 'PLAYOFFS'.")
    else:
        # Get unique team names for the specified year and season
        data = load_tidy_data(year, season)
        unique_teams = data['team'].unique()

        # Create a dictionary for team names
        team_name_dict = {team: team for team in unique_teams}

        # Prompt the user to select a team from the dictionary
        selected_team = input(f"Select a team from the following: {', '.join(unique_teams)}: ")

        # Load the rink image
        rink_image_path = "IFT6758B-Project-B10/Milestone1/visualizations/nhl_rink.png"
        rink_image = load_rink_image(rink_image_path)

        # Create the shot map for the selected team
        fig = create_shot_map_for_team(data, rink_image, selected_team)
        # Display the interactive plot
        fig.show()
        """app = Dash()
        app.layout = html.Div([
            dcc.Graph(figure=fig)
            app.run_server(debug=True, use_reloader=False)
        ])"""

