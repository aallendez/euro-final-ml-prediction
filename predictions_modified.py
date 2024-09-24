import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import statsmodels.api as sm

import os

os.system("clear")

print("\n\n\n\n")

# Load the dataset
euro_matches = pd.read_csv('euro_matches.csv')
print(euro_matches.head(10))

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

print(f"Cross-Validation Accuracy: {cross_val_accuracy:.2f}")
print(f"Accuracy on Semi-finals: {accuracy:.2f}")

# Example stats for Spain and England
spain_stats = {
    'avg_goals_scored': euro_matches[euro_matches['team1'] == 'Spain']['score_ft_team1'].mean(),
    'avg_goals_conceded': euro_matches[euro_matches['team1'] == 'Spain']['score_ft_team2'].mean()
}

england_stats = {
    'avg_goals_scored': euro_matches[euro_matches['team1'] == 'England']['score_ft_team1'].mean(),
    'avg_goals_conceded': euro_matches[euro_matches['team1'] == 'England']['score_ft_team2'].mean()
}

# Predict the winner for the final match (example teams)
final_match_data = pd.DataFrame({
    'avg_goals_scored_team1': [spain_stats['avg_goals_scored']],
    'avg_goals_conceded_team1': [spain_stats['avg_goals_conceded']],
    'avg_goals_scored_team2': [england_stats['avg_goals_scored']],
    'avg_goals_conceded_team2': [england_stats['avg_goals_conceded']]
})

# Predict the winner for the final
final_winner_pred = log_reg.predict(final_match_data)
predicted_final_winner = 'Spain' if final_winner_pred[0] == 1 else 'England'

# Add function to predict the score of the game
def predict_scores(X_train, y_train_score1, y_train_score2, X_test):
    # Add constant to the features
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # Train a Linear Regression model for predicting the scores
    lin_reg1 = sm.OLS(y_train_score1, X_train_const).fit()
    lin_reg2 = sm.OLS(y_train_score2, X_train_const).fit()
    
    # Predict the scores for the test data
    score1_pred = lin_reg1.predict(X_test_const)
    score2_pred = lin_reg2.predict(X_test_const)
    
    return score1_pred, score2_pred

# Prepare the targets for the score predictions
y_train_score1 = train_data['score_ft_team1']
y_train_score2 = train_data['score_ft_team2']

# Predict the scores for the semi-finals
predicted_scores_team1, predicted_scores_team2 = predict_scores(X_train, y_train_score1, y_train_score2, X_test)

# Evaluate the score predictions
mse_team1 = mean_squared_error(test_data['score_ft_team1'], predicted_scores_team1)
mse_team2 = mean_squared_error(test_data['score_ft_team2'], predicted_scores_team2)

print(f"Mean Squared Error for Team 1 Scores: {mse_team1:.2f}")
print(f"Mean Squared Error for Team 2 Scores: {mse_team2:.2f}")

# Predict the score for the final match
final_score1_pred, final_score2_pred = predict_scores(X_train, y_train_score1, y_train_score2, final_match_data)
print(f"\n\nPredicted Score for the Final: Spain {final_score1_pred[0]:.2f} - England {final_score2_pred[0]:.2f}")
print(f"Predicted Winner for the Final: {predicted_final_winner}")

print("\n\n\n\n")

