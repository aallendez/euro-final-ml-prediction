import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

os.system("clear")

# Load data
matches = pd.read_csv('euro_matches.csv')
matches.drop(['date', 'time', 'group', 'team1_code', 'team2_code', 'match_number', 'score_ht_team1', 'score_ht_team2'], axis=1, inplace=True)

# Filter matches involving Spain and England
spain_england_matches = matches[(matches['team1'].isin(['Spain', 'England'])) | (matches['team2'].isin(['Spain', 'England']))]

# One-hot encode 'team1' and 'team2'
matches_encoded = pd.get_dummies(spain_england_matches, columns=['team1', 'team2'])

# Create features and target variables with one-hot encoding
features_encoded = matches_encoded[['score_ft_team1', 'score_ft_team2',
                                    'team1_Spain', 'team1_England', 
                                    'team2_Spain', 'team2_England']]

# Adding goals scored and conceded
features_encoded['goals_scored_spain'] = features_encoded.apply(lambda row: row['score_ft_team1'] if row['team1_Spain'] == 1 else (row['score_ft_team2'] if row['team2_Spain'] == 1 else 0), axis=1)
features_encoded['goals_conceded_spain'] = features_encoded.apply(lambda row: row['score_ft_team2'] if row['team1_Spain'] == 1 else (row['score_ft_team1'] if row['team2_Spain'] == 1 else 0), axis=1)
features_encoded['goals_scored_england'] = features_encoded.apply(lambda row: row['score_ft_team1'] if row['team1_England'] == 1 else (row['score_ft_team2'] if row['team2_England'] == 1 else 0), axis=1)
features_encoded['goals_conceded_england'] = features_encoded.apply(lambda row: row['score_ft_team2'] if row['team1_England'] == 1 else (row['score_ft_team1'] if row['team2_England'] == 1 else 0), axis=1)

os.system("clear")

# Display the first few rows of the dataframe with encoded teams
print(features_encoded.head(10))

# # Check for NaN values
# print(features_encoded.isna().sum())

# # Fill NaN values with 0
# features_encoded = features_encoded.fillna(0)

# # Features for Spain and England
# X_spain_encoded = features_encoded[['goals_scored_spain', 'goals_conceded_england']]
# y_spain_encoded = features_encoded['goals_scored_spain']
# X_england_encoded = features_encoded[['goals_scored_england', 'goals_conceded_spain']]
# y_england_encoded = features_encoded['goals_scored_england']

# # Train the model for Spain
# model_spain_encoded = LinearRegression()
# model_spain_encoded.fit(X_spain_encoded, y_spain_encoded)

# # Train the model for England
# model_england_encoded = LinearRegression()
# model_england_encoded.fit(X_england_encoded, y_england_encoded)

# # Predict Spain's and England's scores
# spain_predictions = model_spain_encoded.predict(X_spain_encoded)
# england_predictions = model_england_encoded.predict(X_england_encoded)

# # Predict Spain's score based on England's goals conceded
# spain_score_encoded = model_spain_encoded.predict([[features_encoded['goals_scored_spain'].mean(), features_encoded['goals_conceded_england'].mean()]])
# # Predict England's score based on Spain's goals conceded
# england_score_encoded = model_england_encoded.predict([[features_encoded['goals_scored_england'].mean(), features_encoded['goals_conceded_spain'].mean()]])

# predicted_score_encoded = f'Predicted score: Spain {spain_score_encoded[0]:.2f} - England {england_score_encoded[0]:.2f}'
# print(predicted_score_encoded)

# # Plotting the results
# plt.figure(figsize=(14, 6))

# # Plot for Spain
# plt.subplot(1, 2, 1)
# plt.scatter(y_spain_encoded, spain_predictions, color='blue')
# plt.plot([y_spain_encoded.min(), y_spain_encoded.max()], [y_spain_encoded.min(), y_spain_encoded.max()], 'k--', lw=2)
# plt.xlabel('Actual Spain Goals')
# plt.ylabel('Predicted Spain Goals')
# plt.title('Actual vs Predicted Goals for Spain')

# # Plot for England
# plt.subplot(1, 2, 2)
# plt.scatter(y_england_encoded, england_predictions, color='red')
# plt.plot([y_england_encoded.min(), y_england_encoded.max()], [y_england_encoded.min(), y_england_encoded.max()], 'k--', lw=2)
# plt.xlabel('Actual England Goals')
# plt.ylabel('Predicted England Goals')
# plt.title('Actual vs Predicted Goals for England')

# plt.tight_layout()
# plt.show()
