import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
import os

os.system("clear")

df = pd.read_csv('Euro_2024_Matches.csv')

def todays_match():
    return 0


def create_df():
    
    # Delete any duplicated entries
    df.drop_duplicates(inplace=True)
    
    df.drop("group", axis=1, inplace=True)

    print(df.dtypes)

    df["date"] = pd.to_datetime(df["date"])
    df["team1"] = df["team1"].astype("category").cat.codes
    df["team2"] = df["team2"].astype("category").cat.codes
    df["time"] = df["time"].str.replace(":.+", "", regex=True).astype(int)
    
    # Assuming 'team1', 'team2', 'team1_score', 'team2_score' are columns in your dataset
    df['team1_win'] = df['score_ft_team1'] > df['score_ft_team2']
    df['team2_win'] = df['score_ft_team1'] < df['score_ft_team2']
    df['draw'] = df['score_ft_team1'] == df['score_ft_team2']

    

    print(df.head(15))
    
    
    
create_df()

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

train, test = train_test_split(df, test_size=0.2)

print(df.head(15))