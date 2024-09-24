import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

import os

os.system("clear")

euro_matches = pd.read_csv('euro_matches.csv')
print(euro_matches.head(10))

# Inspect column names
print(euro_matches.columns)

# Calculate general performance metrics for each team
team_performance = euro_matches.groupby('team1').agg(
    avg_goals_scored=('score_ft_team1', 'mean'),
    avg_goals_conceded=('score_ft_team2', 'mean')
).reset_index()

# Merge these metrics back into the original dataset for both teams
euro_matches = euro_matches.merge(team_performance, left_on='team1', right_on='team1')
euro_matches = euro_matches.merge(team_performance, left_on='team2', right_on='team1', suffixes=('_team1', '_team2'))

# Select features and target
features = ['avg_goals_scored_team1', 'avg_goals_conceded_team1',
            'avg_goals_scored_team2', 'avg_goals_conceded_team2']
euro_matches['winner'] = (euro_matches['score_ft_team1'] > euro_matches['score_ft_team2']).astype(int)

# Split the data
train_data = euro_matches[euro_matches['round'].isin(['Matchday 1', 'Matchday 2', 'Matchday 3', 'Round of 16', 'Quarter-finals'])]
test_data = euro_matches[euro_matches['round'] == 'Semi-finals']

X_train = train_data[features]
y_train = train_data['winner']
X_test = test_data[features]
y_test = test_data['winner']

# Train a Logistic Regression model for predicting the winner
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Evaluate with cross-validation
cross_val_accuracy = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='accuracy').mean()

# Predict and evaluate the semi-finals using the trained model
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Cross-Validation Accuracy: {cross_val_accuracy * 100:.2f}%")
print(f"Semi-Finals Accuracy: {accuracy * 100:.2f}%")

# Extract the relevant metrics for Spain and England
spain_stats = team_performance[team_performance['team1'] == 'Spain'].iloc[0]
england_stats = team_performance[team_performance['team1'] == 'England'].iloc[0]

# Create the feature set for the final match
final_match_data = pd.DataFrame({
    'avg_goals_scored_team1': [spain_stats['avg_goals_scored']],
    'avg_goals_conceded_team1': [spain_stats['avg_goals_conceded']],
    'avg_goals_scored_team2': [england_stats['avg_goals_scored']],
    'avg_goals_conceded_team2': [england_stats['avg_goals_conceded']]
})

# Predict the winner for the final
final_winner_pred = log_reg.predict(final_match_data)
predicted_final_winner = 'Spain' if final_winner_pred[0] == 1 else 'England'

print(f"Predicted Winner for the Final: {predicted_final_winner}")