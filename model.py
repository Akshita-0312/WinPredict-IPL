# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import GradientBoostingClassifier
# import pickle

# df = pd.read_csv("matches.csv")

# df = df[['team1','team2','toss_winner','toss_decision','venue','winner']]
# df.dropna(inplace=True)

# # 👉 Add synthetic weather data (since dataset doesn't have it)
# import random
# weather_options = ['Hot', 'Humid', 'Dew', 'Cloudy']
# df['weather'] = [random.choice(weather_options) for _ in range(len(df))]

# # Encode
# encoders = {}
# for col in df.columns:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     encoders[col] = le

# X = df.drop('winner', axis=1)
# y = df['winner']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = GradientBoostingClassifier()
# model.fit(X_train, y_train)

# pickle.dump(model, open("model.pkl", "wb"))
# pickle.dump(encoders, open("encoders.pkl", "wb"))











import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Load data
matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

# Keep required columns
matches = matches[['id', 'winner']]
deliveries = deliveries.merge(matches, left_on='match_id', right_on='id')

# Only 2nd innings (chasing team)
deliveries = deliveries[deliveries['inning'] == 2]

# Target runs
total_runs = deliveries.groupby('match_id')['total_runs'].sum().reset_index()
total_runs.rename(columns={'total_runs': 'total_runs_x'}, inplace=True)
deliveries = deliveries.merge(total_runs, on='match_id')

# ================= FEATURE ENGINEERING =================

# Current score
deliveries['current_score'] = deliveries.groupby('match_id')['total_runs'].cumsum()

# Runs left
deliveries['runs_left'] = deliveries['total_runs_x'] - deliveries['current_score']

# Balls left
deliveries['balls_left'] = 120 - (deliveries['over'] * 6 + deliveries['ball'])

# Wickets
deliveries['player_dismissed'] = deliveries['player_dismissed'].notna().astype(int)
deliveries['wickets'] = deliveries.groupby('match_id')['player_dismissed'].cumsum()
deliveries['wickets_left'] = 10 - deliveries['wickets']

# Run rates
deliveries['crr'] = deliveries['current_score'] / ((deliveries['over'] * 6 + deliveries['ball']) / 6)
deliveries['rrr'] = (deliveries['runs_left'] * 6) / deliveries['balls_left']

# Result
deliveries['result'] = (deliveries['batting_team'] == deliveries['winner']).astype(int)

# Clean
deliveries.replace([np.inf, -np.inf], np.nan, inplace=True)
deliveries.dropna(inplace=True)

# ================= FEATURES =================

features = deliveries[['batting_team','bowling_team','runs_left',
                       'balls_left','wickets_left','total_runs_x','crr','rrr']]

target = deliveries['result']

# One-hot encoding
features = pd.get_dummies(features, columns=['batting_team','bowling_team'])

# Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model
model = GradientBoostingClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Save
pickle.dump(model, open("live_model.pkl","wb"))
pickle.dump(features.columns, open("columns.pkl","wb"))

print("✅ Model trained successfully")