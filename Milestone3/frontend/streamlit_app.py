import streamlit as st
import time
import plotly.graph_objects as go
import plotly.express as px
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
    version = st.text_input("Version")

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
        if game_data['status_code'] != 200:
            alert = st.error('Invalid Game ID. Please try again!')
            time.sleep(3)
            alert.empty()
        elif game_data['data'].empty:
            alert = st.warning('You are early! This game has not happened yet. Please try again later!')
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
        feature_list.insert(0, 'team')

        df = data['data'].set_index('eventType')
        df['team'] = df.apply(lambda row: data['home_team_name'] if row['team'] == data['home_team_code'] else data['away_team_name'], axis=1)
        st.dataframe(df[feature_list])

with st.container():
    if data:
        st.title('Shot Location Heatmap')

        df = data['data']
        df['x_coordinate'] = df.apply(lambda row: -1* row['x_coordinate'] if row['goal_location'] == 'left' else row['x_coordinate'], axis=1)

        fig = go.Figure()

        fig.add_trace(go.Scattergl(
            x=df[df['team'] == data['home_team_code']]['x_coordinate'],
            y=df[df['team'] == data['home_team_code']]['y_coordinate'],
            mode='markers',
            marker=dict(
                size=14,
                opacity=0.6,
                colorscale='Viridis',
                color='blue'
            ),
            name=data['home_team_name'],
        ))

        fig.add_trace(go.Scattergl(
            x=df[df['team'] == data['away_team_code']]['x_coordinate'],
            y=df[df['team'] == data['away_team_code']]['y_coordinate'],
            mode='markers',
            marker=dict(
                size=14,
                opacity=0.6,
                colorscale='Viridis',
                color='red'
            ),
            name=data['away_team_name'],
        ))

        fig.update_layout(
            title='Shot Location Heatmap',
            xaxis=dict(title='X Coordinate'),
            yaxis=dict(title='Y Coordinate'),
        )

        st.plotly_chart(fig)

        st.write("""
            The interactive combined shot location map aimed at visualizing the various shot location trend for a given team.
            X-coordinates of the shots were transformed so that for all the shots the goal was assumed to be on the right side. 
            This was done to establish a consistent frame of reference.
            Finally the data was plotted to visualize the trend.
        """)

with st.container():
    if data:
        shot_events = data['data']

        sum_goal_probs = shot_events.groupby(['shotType', 'team'])['goal_prob_prediction'].sum().reset_index()

        sum_goal_probs['team'] = sum_goal_probs['team'].replace(data['home_team_code'], data['home_team_name'])
        sum_goal_probs['team'] = sum_goal_probs['team'].replace(data['away_team_code'], data['away_team_name'])

        st.title('Expected Goals for Shot Types Grouped by Teams')

        st.subheader('Visualization for Expected Goals')
        fig = px.area(sum_goal_probs, x='shotType', y='goal_prob_prediction', color='team',
                      labels={'goal_prob_prediction': 'Expected Goals', 'shotType': 'Shot Type', 'team': 'Team'},
                      title='Expected Goals for Shot Types Grouped by Teams',
                      line_shape='linear',
                      hover_data={'goal_prob_prediction': ':.2f', 'shotType': False, 'team': False}
                      )

        st.plotly_chart(fig)

        st.write("""
            This visualization aimed at finding the most impactful shot type i.e. the shot type that leads to highest number of expected goals for a given team.
            For this, the data was grouped by shot type and team and then the individual goal probabilities were summed. The transformed data was then plotted 
            on an area chart to visualize the trend.
        """)