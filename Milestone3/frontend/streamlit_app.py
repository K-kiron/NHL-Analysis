import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import sys
sys.path.append('../client')

from game_client import GameClient
from serving_client import ServingClient

st.title("Hockey Visualization App")

serving = ServingClient()
game = GameClient()

with (st.sidebar):
    st.header("Model Details")

    ws_name = st.text_input("Workspace", "ift6758b-project-b10")
    model = st.text_input("Model")
    version = st.text_input("Choose Input 3")

    if st.button("Get model"):
        res = serving.download_registry_model(workspace=ws_name, model=model, version=version)

        if res.status_code == 200:
            alert = st.success("Model changed successfully!")
            time.sleep(3)
            alert.empty()
        else:
            alert = st.error('Model details are incorrect. Please try again!')
            time.sleep(3)
            alert.empty()

with st.container():
    data = None
    game_id = st.text_input("Game ID")
    if st.button("Ping game"):
        game_data = game.ping_game(game_id)
        if game_data['status_code'] == 400:
            alert = st.error('Invalid Game ID. Please try again!')
            time.sleep(3)
            alert.empty()
        else:
            game_preds = serving.predict(game_data['data']).json()
            game_data['data']['goal_prob_prediction'] = game_preds['MODEL predictions']
            game_data['features_used'] = game_preds['features_used']

            data = game_data

with st.container():
    if data:
        st.header(f"Game {game_id}: {game_data['home_team_name']} vs {game_data['away_team_name']}")
        xG = data['data'].groupby('team')['goal_prob_prediction'].sum()
        xG_home_team = round(xG[data['home_team_code']], 2)
        xG_away_team = round(xG[data['away_team_code']], 2)

        delta_home_team = round(data['home_team_score'] - xG_home_team, 2)
        delta_away_team = round(data['away_team_score'] - xG_away_team, 2)

        col1, col2 = st.columns(2)

        col1.metric(label=f"{game_data['home_team_name']} xG (actual)",
                    value=f"{xG_home_team} ({data['home_team_score']})",
                    delta=delta_home_team)

        col2.metric(label=f"{game_data['away_team_name']} xG (actual)",
                    value=f"{xG_away_team} ({data['away_team_score']})",
                    delta=delta_away_team)

with st.container():
    if data:
        st.header(f"Data used for predictions (and predictions)")
        feature_list = data['features_used']
        feature_list.append('goal_prob_prediction')

        df = data['data'].set_index('eventType')
        st.dataframe(df[feature_list])

@st.cache
def update_filtered_df(selected_player, data):
    filtered_df = data['data'][data['data']['shooter'] == selected_player]
    return filtered_df

with st.container():
    if data:
        if 'selected_team' not in st.session_state:
            st.session_state.selected_team = data['home_team_name']

        if 'selected_player' not in st.session_state:
            st.session_state.selected_player = data['data'].loc[data['data']['team'] == data['home_team_code'], 'shooter'].iloc[0]

        if 'filtered_df' not in st.session_state:
            st.session_state.filtered_df = data['data'][data['data']['shooter'] == st.session_state.selected_player]

        # Streamlit app
        st.title('Shot Location Heatmap')

        # col1, col2 = st.columns(2)
        #
        # # Sidebar for user input
        # st.session_state.selected_team = col1.selectbox('Select Team:', [data['home_team_name'], data['away_team_name']],
        #                                key='team_selector')
        # code = data['away_team_code'] if st.session_state.selected_team == data['away_team_name'] else data['home_team_code']
        #
        # st.session_state.selected_player = col2.selectbox('Select Player:',
        #                                  data['data'].loc[data['data']['team'] == code, 'shooter'].unique(),
        #                                  key='player_selector')

        # Update filtered_df based on user input
        filtered_df_home = data['data'][data['data']['team'] == data['home_team_code']]
        filtered_df_away = data['data'][data['data']['team'] == data['away_team_code']]

        # Create a heatmap using Plotly Graph Objects
        fig = go.Figure(go.Scattergl(
            x=filtered_df_home['x_coordinate'],
            y=filtered_df_home['y_coordinate'],
            mode='markers',
            marker=dict(
                size=14,
                opacity=0.6,
                colorscale='Viridis',
                colorbar=dict(title='Intensity'),
            ),
        ))

        # Set axis labels and layout
        fig.update_layout(
            title=f"{data['home_team_name']} Shot Location Heatmap",
            xaxis=dict(title='X Position'),
            yaxis=dict(title='Y Position'),
        )

        # Display the heatmap
        st.plotly_chart(fig)

        # Create a heatmap using Plotly Graph Objects
        fig1 = go.Figure(go.Scattergl(
            x=filtered_df_away['x_coordinate'],
            y=filtered_df_away['y_coordinate'],
            mode='markers',
            marker=dict(
                size=14,
                opacity=0.6,
                colorscale='Viridis',
                colorbar=dict(title='Intensity'),
            ),
        ))

        # Set axis labels and layout
        fig1.update_layout(
            title=f"{data['away_team_name']} Shot Location Heatmap",
            xaxis=dict(title='X Position'),
            yaxis=dict(title='Y Position'),
        )

        # Display the heatmap
        st.plotly_chart(fig1)

        # Additional information about the feature
        st.write("""
            Explore the player location heatmap with our Player Location Heatmap feature. 
            Select a player from the sidebar to visualize the X and Y coordinates of their 
            positions on the ice. This static heatmap leverages the power of Streamlit and 
            Plotly to provide an engaging and informative experience for hockey enthusiasts, 
            allowing you to examine the spatial dynamics of player movements.
        """)