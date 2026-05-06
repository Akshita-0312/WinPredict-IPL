import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="WinPredict IPL",
    layout="centered"
)

plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'

st.title("🏏 WinPredict IPL : IPL Match Winner Prediction")

# ================= UI DESIGN =================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

.stApp {
    background: linear-gradient(to right, #d6e4ff, #e4d6ff);
    font-family: 'Poppins', sans-serif;
}

h1 {
    color: #4b3fbf;
    text-align: center;
}

label {
    font-weight: 600 !important;
    color: #2f2f2f !important;
}

div[data-baseweb="select"],
div[data-baseweb="input"] {
    background-color: #dbe7ff !important;
    border-radius: 10px !important;
}

input {
    background-color: #dbe7ff !important;
    border: none !important;
}

.stButton>button {
    background: linear-gradient(45deg, #5a8dee, #9f6bff);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODELS =================

live_model = pickle.load(open("live_model.pkl", "rb"))

cols = pickle.load(open("columns.pkl", "rb"))

prematch_model = pickle.load(
    open("prematch_model.pkl", "rb")
)

prematch_encoders = pickle.load(
    open("prematch_encoders.pkl", "rb")
)

# ================= LOAD DATA =================

df = pd.read_csv("matches.csv")

df['venue'] = df['venue'].str.strip()

venues = sorted(df['venue'].dropna().unique())

# Automatically fetch teams
teams = sorted(
    list(
        set(df['team1'].unique()) |
        set(df['team2'].unique())
    )
)

# =====================================================
# 🏆 PRE-MATCH PREDICTION
# =====================================================

st.header("🏆 Pre-Match Prediction")

team1 = st.selectbox(
    "Team 1",
    teams
)

team2 = st.selectbox(
    "Team 2",
    [t for t in teams if t != team1]
)

venue_pre = st.selectbox(
    "Venue",
    venues
)

toss_winner = st.selectbox(
    "Toss Winner",
    [team1, team2]
)

toss_decision = st.selectbox(
    "Toss Decision",
    ['bat', 'field']
)

# ================= HEAD TO HEAD =================

h2h_df = df[
    (
        (df['team1'] == team1) &
        (df['team2'] == team2)
    ) |
    (
        (df['team1'] == team2) &
        (df['team2'] == team1)
    )
]

team1_h2h = h2h_df[
    h2h_df['winner'] == team1
].shape[0]

team2_h2h = h2h_df[
    h2h_df['winner'] == team2
].shape[0]

# ================= PRE-MATCH PREDICTION =================

if st.button("🏆 Predict Pre-Match Winner"):

    input_data = pd.DataFrame([{
        'team1': team1,
        'team2': team2,
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'venue': venue_pre,
        'team1_h2h_wins': team1_h2h,
        'team2_h2h_wins': team2_h2h
    }])

    # Encode categorical columns
    for col in [
        'team1',
        'team2',
        'toss_winner',
        'toss_decision',
        'venue'
    ]:

        le = prematch_encoders[col]

        input_data[col] = le.transform(
            input_data[col]
        )

    # Prediction
    prediction = prematch_model.predict(
        input_data
    )[0]

    probabilities = prematch_model.predict_proba(
        input_data
    )[0]

    winner = prematch_encoders[
        'winner'
    ].inverse_transform([prediction])[0]

    win_prob = round(max(probabilities) * 100, 2)

    # ================= RESULT =================

    st.success(
        f"🏆 Predicted Winner: {winner}"
    )

    st.info(
        f"📊 Winning Probability: {win_prob}%"
    )

    # ================= CHART =================

    st.subheader("📈 Team Winning Chances")

    fig0, ax0 = plt.subplots(figsize=(6,4))

    ax0.bar(
        [team1, team2],
        [probabilities[0]*100,
         probabilities[1]*100],
        color=['#e74c3c', '#3498db']
    )

    plt.xticks(rotation=10)

    st.pyplot(fig0)

    # ================= HEAD TO HEAD =================

    st.subheader("🥧 Head-to-Head")

    st.markdown(f"""
    <div style="text-align:center; font-size:16px;">
    Total Matches: <b>{h2h_df.shape[0]}</b><br>
    {team1} Wins: <b>{team1_h2h}</b><br>
    {team2} Wins: <b>{team2_h2h}</b>
    </div>
    """, unsafe_allow_html=True)

    fig_h2h, ax_h2h = plt.subplots(figsize=(6,4))

    values = [
        team1_h2h if team1_h2h > 0 else 0.1,
        team2_h2h if team2_h2h > 0 else 0.1
    ]

    ax_h2h.pie(
        values,
        labels=[team1, team2],
        autopct='%1.1f%%',
        colors=['#f39c12','#2ecc71'],
        startangle=90
    )

    ax_h2h.set_aspect('equal')

    st.pyplot(fig_h2h)

# =====================================================
# 🔥 LIVE MATCH PREDICTION
# =====================================================

st.header("🔥 Live Match Prediction")

batting_team = st.selectbox(
    "Batting Team",
    teams
)

bowling_team = st.selectbox(
    "Bowling Team",
    [t for t in teams if t != batting_team]
)

venue = st.selectbox(
    "🏟️ Match Venue",
    venues
)

target = st.number_input(
    "Target",
    50,
    300,
    160
)

score = st.number_input(
    "Current Score",
    0,
    300,
    80
)

overs = st.number_input(
    "Overs",
    0.0,
    20.0,
    10.0,
    step=0.1
)

wickets = st.number_input(
    "Wickets",
    0,
    10,
    2
)

# ================= CALCULATIONS =================

runs_left = target - score

balls_left = 120 - int(overs * 6)

wickets_left = 10 - wickets

crr = score / overs if overs > 0 else 0

rrr = (
    runs_left * 6 / balls_left
) if balls_left > 0 else 0

# ================= LIVE PREDICTION =================

if st.button("🔮 Predict Match Outcome"):

    input_dict = {
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets_left': wickets_left,
        'total_runs_x': target,
        'crr': crr,
        'rrr': rrr
    }

    for col in cols:
        if col not in input_dict:
            input_dict[col] = 0

    input_dict[
        f'batting_team_{batting_team}'
    ] = 1

    input_dict[
        f'bowling_team_{bowling_team}'
    ] = 1

    input_df = pd.DataFrame(
        [input_dict]
    )[cols]

    proba = live_model.predict_proba(
        input_df
    )[0]

    win = round(proba[1] * 100, 2)

    lose = round(proba[0] * 100, 2)

    # ================= RESULT =================

    st.subheader("🎯 Match Insight")

    if win > lose:

        st.success(
            f"🔥 {batting_team} leads with {win}% chance"
        )

        st.progress(int(win))

    else:

        st.error(
            f"⚡ {bowling_team} leads with {lose}% chance"
        )

        st.progress(int(lose))

    # ================= WIN PROBABILITY =================

    st.subheader("📊 Win Probability")

    fig1, ax1 = plt.subplots(figsize=(6,4))

    ax1.bar(
        [batting_team, bowling_team],
        [win, lose],
        color=['#e74c3c', '#3498db']
    )

    plt.xticks(rotation=10)

    st.pyplot(fig1)

    # ================= HEAD TO HEAD =================

    st.subheader("🥧 Head-to-Head")

    h2h_df_live = df[
        (
            (df['team1'] == batting_team) &
            (df['team2'] == bowling_team)
        ) |
        (
            (df['team1'] == bowling_team) &
            (df['team2'] == batting_team)
        )
    ]

    if h2h_df_live.shape[0] > 0:

        total_matches = h2h_df_live.shape[0]

        h2h_wins = h2h_df_live[
            'winner'
        ].value_counts()

        h2h_wins = h2h_wins.reindex(
            [batting_team, bowling_team],
            fill_value=0
        )

        st.markdown(f"""
        <div style="text-align:center; font-size:16px;">
        Total Matches: <b>{total_matches}</b><br>
        {batting_team} Wins:
        <b>{h2h_wins[batting_team]}</b><br>

        {bowling_team} Wins:
        <b>{h2h_wins[bowling_team]}</b>
        </div>
        """, unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(6,4))

        values = [
            v if v > 0 else 0.1
            for v in h2h_wins.values
        ]

        ax2.pie(
            values,
            labels=h2h_wins.index,
            autopct='%1.1f%%',
            colors=['#f39c12','#2ecc71'],
            startangle=90
        )

        ax2.set_aspect('equal')

        st.pyplot(fig2)

    else:
        st.warning(
            "No head-to-head matches found."
        )

    # ================= MATCH OVERVIEW =================

    st.subheader("📊 Match Overview")

    fig4, ax4 = plt.subplots(figsize=(6,4))

    colors = [
        '#ff6b6b',
        '#4ecdc4',
        '#ffe66d',
        '#5f27cd',
        '#1dd1a1'
    ]

    ax4.bar(
        [
            "Runs Left",
            "Balls Left",
            "Wickets",
            "CRR",
            "RRR"
        ],
        [
            runs_left,
            balls_left,
            wickets_left,
            crr,
            rrr
        ],
        color=colors
    )

    plt.xticks(rotation=10)

    st.pyplot(fig4)

    # ================= MATCH SITUATION =================

    st.subheader("📋 Match Situation")

    st.markdown(f"""
    <div style="text-align:center; font-size:18px;">
    Runs Left: <b>{runs_left}</b><br>
    Balls Left: <b>{balls_left}</b><br>
    Wickets Left: <b>{wickets_left}</b><br>
    CRR: <b>{crr:.2f}</b><br>
    RRR: <b>{rrr:.2f}</b>
    </div>
    """, unsafe_allow_html=True)

    # ================= VENUE HEAD TO HEAD =================

    st.subheader("🏟️ Venue Head-to-Head")

    venue_df = df[
        (df['venue'] == venue) &
        (
            (
                (df['team1'] == batting_team) &
                (df['team2'] == bowling_team)
            ) |
            (
                (df['team1'] == bowling_team) &
                (df['team2'] == batting_team)
            )
        )
    ]

    if venue_df.shape[0] > 0:

        total_matches_v = venue_df.shape[0]

        venue_wins = venue_df[
            'winner'
        ].value_counts()

        venue_wins = venue_wins.reindex(
            [batting_team, bowling_team],
            fill_value=0
        )

        st.markdown(f"""
        <div style="text-align:center; font-size:16px;">
        Total Matches: <b>{total_matches_v}</b><br>
        {batting_team} Wins:
        <b>{venue_wins[batting_team]}</b><br>

        {bowling_team} Wins:
        <b>{venue_wins[bowling_team]}</b>
        </div>
        """, unsafe_allow_html=True)

        fig5, ax5 = plt.subplots(figsize=(6,4))

        ax5.bar(
            venue_wins.index,
            venue_wins.values,
            color=['#c0392b','#27ae60']
        )

        plt.xticks(rotation=10)

        st.pyplot(fig5)

    else:

        st.warning(
            "No matches between these teams at this venue."
        )