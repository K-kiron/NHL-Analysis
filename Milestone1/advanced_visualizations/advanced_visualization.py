import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Import the tidy_data function from '../features/tidy_data.py'
from features.tidy_data import tidy_data  

# Define the data path
data_path = '../data/IFT6758_Data'

def create_shot_map(team_name, year, season, game_data):
    # Use the tidy_data function to load the sorted data
    sorted_data = tidy_data(data_path, year, season, game_data)

    # Filter sorted_data to get shots for the specified team
    shots = sorted_data[(sorted_data['team'] == team_name)]

    # Load the ice rink image from the current directory
    current_directory = os.path.dirname(os.path.realpath(__file__))
    ice_rink_image_path = os.path.join(current_directory, 'nhl_rink.png')

    # Create a figure with the ice rink image as the background
    img = plt.imread(ice_rink_image_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, extent=[-100, 100, -42.5, 42.5])

    # Plot shots on the ice rink
    ax.scatter(shots['x_coordinate'], shots['y_coordinate'], color='red', alpha=0.6, s=30)

    # Customize the plot
    ax.set_xlim(-100, 100)
    ax.set_ylim(-42.5, 42.5)
    ax.set_xlabel('Feet')
    ax.set_ylabel('Feet')
    ax.set_title(f'{team_name} Shot Map - {year}-{season}')

    # Save the plot as an HTML file
    html_file_path = f'{team_name}_{year}_{season}_shot_map.html'
    plt.savefig(html_file_path, format='html')

    # Display the interactive plot using Plotly
    fig = go.Figure()
    for index, shot in shots.iterrows():
        fig.add_trace(go.Scatter(x=[shot['x_coordinate']], y=[shot['y_coordinate']], mode='markers', marker=dict(size=10, color='red')))
    fig.update_layout(
        title=f'{team_name} Shot Map - {year}-{season}',
        xaxis=dict(title='Feet'),
        yaxis=dict(title='Feet'),
        showlegend=False
    )

    # Save the Plotly figure as an HTML file
    plotly_html_file_path = f'{team_name}_{year}_{season}_plotly_shot_map.html'
    fig.write_html(plotly_html_file_path)

    # Show the Plotly figure
    fig.show()

# Example usage:
# create_shot_map('Colorado Avalanche', 2016, 'regular', id)

