import json
import os
import glob
import pandas as pd
import numpy as np
import math

def feature_eng2(df):
    df_eng2 = df
    df_eng2[['minutes', 'seconds']] = df_eng2['periodTime'].str.split(':', expand=True)
    df_eng2['gameSeconds'] = df['period']*(df_eng2['minutes'].astype(int) * 60 + df_eng2['seconds'].astype(int))
    df_eng2 = df_eng2.drop(columns=['minutes', 'seconds'])

    df_copy = df
    new_df = df_copy.shift(fill_value=np.nan)
    new_df.columns = df.columns

    new_df.iloc[0, :] = np.nan
    new_df.iloc[:, 0] = np.nan

    df_copy = new_df
    df_eng2['LastEventType'] = df_copy['eventType']
    df_eng2['Last_x_coordinate'] = df_copy['x_coordinate']
    df_eng2['Last_y_coordinate'] = df_copy['y_coordinate']
    df_eng2['Last_gameSeconds'] = df_copy['gameSeconds']
    df_eng2['Last_period'] = df_copy['period']
    df_eng2['DistanceLastEvent'] = np.sqrt((df_eng2['Last_x_coordinate']-df_eng2['x_coordinate'])**2+(df_eng2['Last_y_coordinate']-df_eng2['y_coordinate'])**2)
    df_eng2['Rebound'] = df_eng2['LastEventType'] == 'Shot'
    df_eng2['LastShotAngle'] = df_copy['shotAngle']
    df_eng2['changeShotAngle'] = df_eng2['LastShotAngle']+df_eng2['shotAngle']
    df_eng2['timeFromLastEvent'] = df_eng2['gameSeconds']-df_eng2['Last_gameSeconds']
    df_eng2['speed'] = df_eng2['DistanceLastEvent']/df_eng2['timeFromLastEvent']
    
    return df_eng2

def feature_eng2_cleaned(df):
    df_eng2 = df
    df_eng2[['minutes', 'seconds']] = df_eng2['periodTime'].str.split(':', expand=True)
    df_eng2['gameSeconds'] = df['period']*(df_eng2['minutes'].astype(int) * 60 + df_eng2['seconds'].astype(int))
    df_eng2 = df_eng2.drop(columns=['minutes', 'seconds'])
    

    df_copy = df
    new_df = df_copy.shift(fill_value=np.nan)
    new_df.columns = df.columns

    new_df.iloc[0, :] = np.nan
    new_df.iloc[:, 0] = np.nan

    df_copy = new_df
    df_eng2['LastEventType'] = df_copy['eventType']
    df_eng2['Last_x_coordinate'] = df_copy['x_coordinate']
    df_eng2['Last_y_coordinate'] = df_copy['y_coordinate']
    df_eng2['Last_gameSeconds'] = df_copy['gameSeconds']
    df_eng2['Last_period'] = df_copy['period']
    df_eng2['DistanceLastEvent'] = np.sqrt((df_eng2['Last_x_coordinate']-df_eng2['x_coordinate'])**2+(df_eng2['Last_y_coordinate']-df_eng2['y_coordinate'])**2)
    df_eng2['Rebound'] = df_eng2['LastEventType'] == 'Shot'
    df_eng2['LastShotAngle'] = df_copy['shotAngle']
    df_eng2['changeShotAngle'] = df_eng2['LastShotAngle']+df_eng2['shotAngle']
    df_eng2['timeFromLastEvent'] = df_eng2['gameSeconds']-df_eng2['Last_gameSeconds']
    df_eng2['speed'] = df_eng2['DistanceLastEvent']/df_eng2['timeFromLastEvent']
    
    return df_eng2[['gameSeconds','period','x_coordinate','y_coordinate','shotDistance','shotAngle','shotType','LastEventType','Last_x_coordinate','Last_y_coordinate','timeFromLastEvent','DistanceLastEvent','Rebound','changeShotAngle','speed']]

