import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ================= LOAD DATA =================

df = pd.read_csv("matches.csv")

# ================= REQUIRED COLUMNS =================

df = df[['team1','team2','toss_winner',
         'toss_decision','venue','winner']]

df.dropna(inplace=True)

# ================= HEAD TO HEAD FEATURES =================

team1_wins = []
team2_wins = []

for i,row in df.iterrows():

    t1 = row['team1']
    t2 = row['team2']

    previous = df.iloc[:i]

    t1_count = previous[
        (
            (previous['team1']==t1) &
            (previous['team2']==t2) &
            (previous['winner']==t1)
        ) |
        (
            (previous['team1']==t2) &
            (previous['team2']==t1) &
            (previous['winner']==t1)
        )
    ].shape[0]

    t2_count = previous[
        (
            (previous['team1']==t1) &
            (previous['team2']==t2) &
            (previous['winner']==t2)
        ) |
        (
            (previous['team1']==t2) &
            (previous['team2']==t1) &
            (previous['winner']==t2)
        )
    ].shape[0]

    team1_wins.append(t1_count)
    team2_wins.append(t2_count)

df['team1_h2h_wins'] = team1_wins
df['team2_h2h_wins'] = team2_wins

# ================= ENCODING =================

encoders = {}

for col in ['team1','team2','toss_winner',
            'toss_decision','venue','winner']:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

# ================= FEATURES =================

X = df.drop('winner', axis=1)

y = df['winner']

# ================= SPLIT =================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ================= MODEL =================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ================= SAVE =================

pickle.dump(model, open("prematch_model.pkl","wb"))

pickle.dump(encoders, open("prematch_encoders.pkl","wb"))

print("✅ Pre-match model trained successfully")