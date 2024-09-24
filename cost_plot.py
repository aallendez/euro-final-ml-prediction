import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

print(euro_matches.head(15))

X_train = train_data[features].values
y_train = train_data['winner'].values
X_test = test_data[features].values
y_test = test_data['winner'].values

# Add intercept to the features
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Gradient Descent for Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = (-1/m) * (y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h)))
    return cost

def gradient_descent_logistic(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    
    for i in range(num_iters):
        h = sigmoid(X.dot(theta))
        gradient = (1/m) * X.T.dot(h - y)
        theta -= alpha * gradient
        J_history.append(compute_cost_logistic(X, y, theta))
    
    return theta, J_history

# Initialize parameters
theta_logistic = np.zeros(X_train.shape[1])
alpha = 0.01
num_iters = 1000

# Perform gradient descent
theta_logistic, J_history_logistic = gradient_descent_logistic(X_train, y_train, theta_logistic, alpha, num_iters)

# Plot the cost function history
plt.figure(figsize=(10, 6))
plt.plot(J_history_logistic)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent - Logistic Regression')
plt.show()

# Gradient Descent for Linear Regression
def compute_cost_linear(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    cost = (1/(2*m)) * np.sum((h - y) ** 2)
    return cost

def gradient_descent_linear(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    
    for i in range(num_iters):
        h = X.dot(theta)
        gradient = (1/m) * X.T.dot(h - y)
        theta -= alpha * gradient
        J_history.append(compute_cost_linear(X, y, theta))
    
    return theta, J_history

# Prepare the targets for the score predictions
y_train_score1 = train_data['score_ft_team1'].values
y_train_score2 = train_data['score_ft_team2'].values

# Initialize parameters
theta_linear1 = np.zeros(X_train.shape[1])
theta_linear2 = np.zeros(X_train.shape[1])

# Perform gradient descent
theta_linear1, J_history_linear1 = gradient_descent_linear(X_train, y_train_score1, theta_linear1, alpha, num_iters)
theta_linear2, J_history_linear2 = gradient_descent_linear(X_train, y_train_score2, theta_linear2, alpha, num_iters)

# Plot the cost function history for both score predictions
plt.figure(figsize=(10, 6))
plt.plot(J_history_linear1, label='Team 1 Score')
plt.plot(J_history_linear2, label='Team 2 Score')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent - Linear Regression')
plt.legend()
plt.show()
