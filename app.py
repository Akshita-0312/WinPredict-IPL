# import streamlit as st
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt

# # Load model & encoders
# model = pickle.load(open("model.pkl", "rb"))
# encoders = pickle.load(open("encoders.pkl", "rb"))

# # Load dataset
# df = pd.read_csv("matches.csv")

# st.title("🏏 Criclytics - Smart IPL Predictor")

# # Inputs
# teams = sorted(df['team1'].unique())
# venues = sorted(df['venue'].unique())

# team1 = st.selectbox("Team 1", teams)
# team2 = st.selectbox("Team 2", teams)

# if team1 == team2:
#     st.error("⚠️ Please select two different teams")

# toss_winner = st.selectbox("Toss Winner", teams)
# toss_decision = st.selectbox("Toss Decision", ['bat', 'field'])
# venue = st.selectbox("Venue", venues)

# weather = st.selectbox("Weather Condition", ['Hot', 'Humid', 'Dew', 'Cloudy'])

# def encode(val, col):
#     return encoders[col].transform([val])[0]

# # ================== PREDICTION ==================
# if st.button("Predict Winner"):

#     if team1 == team2:
#         st.stop()

#     input_data = [[
#         encode(team1, 'team1'),
#         encode(team2, 'team2'),
#         encode(toss_winner, 'toss_winner'),
#         encode(toss_decision, 'toss_decision'),
#         encode(venue, 'venue'),
#         encode(weather, 'weather')
#     ]]
    
#     proba = model.predict_proba(input_data)[0]
#     classes = encoders['winner'].inverse_transform(range(len(proba)))

#     result_df = pd.DataFrame({
#         "Team": classes,
#         "Win Probability": proba * 100
#     })

#     # Filter only selected teams
#     result_df = result_df[result_df['Team'].isin([team1, team2])]

#     # Normalize
#     total = result_df["Win Probability"].sum()
#     result_df["Win Probability"] = (result_df["Win Probability"] / total) * 100

#     result_df = result_df.sort_values(by="Win Probability", ascending=False)

#     # Result
#     st.subheader("🏆 Prediction Result")
#     st.dataframe(result_df)

#     st.subheader("📊 Win Probability Graph")
#     st.bar_chart(result_df.set_index("Team"))

#     # ================== HISTORICAL ANALYSIS ==================
#     st.subheader("📈 Historical Analysis")

#     # Filter head-to-head matches
#     h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) |
#              ((df['team1'] == team2) & (df['team2'] == team1))]

#     total_matches = h2h.shape[0]
#     st.write(f"📍 Total Matches Played: {total_matches}")

#     # 🥧 Overall Wins Pie Chart
#     overall_wins = h2h['winner'].value_counts()

#     st.write("### 🥧 Overall Head-to-Head Wins")
#     fig1, ax1 = plt.subplots()
#     ax1.pie(overall_wins, labels=overall_wins.index, autopct='%1.1f%%')
#     ax1.set_title("Overall Wins Distribution")
#     st.pyplot(fig1)

#     # 🏟️ Venue-based analysis
#     venue_h2h = h2h[h2h['venue'] == venue]

#     st.write(f"### 🏟️ Wins at {venue}")
#     if venue_h2h.shape[0] > 0:
#         venue_wins = venue_h2h['winner'].value_counts()

#         fig2, ax2 = plt.subplots()
#         ax2.pie(venue_wins, labels=venue_wins.index, autopct='%1.1f%%')
#         ax2.set_title(f"Wins at {venue}")
#         st.pyplot(fig2)
#     else:
#         st.info("No matches played between these teams at this venue.")

#     # 📊 Toss Impact
#     st.write("### 🎯 Toss Decision Impact")

#     toss_analysis = h2h.groupby(['toss_decision', 'winner']).size().unstack(fill_value=0)
#     st.bar_chart(toss_analysis)

#     # ================== EXTRA INSIGHTS ==================
#     st.subheader("📊 Additional Insights")

#     # Matches per venue
#     venue_count = h2h['venue'].value_counts().head(5)
#     st.write("📍 Top Venues Played")
#     st.bar_chart(venue_count)

#     # Toss winners frequency
#     toss_win = h2h['toss_winner'].value_counts()
#     st.write("🪙 Toss Winners Distribution")
#     st.bar_chart(toss_win)

#     # ================== EXPLANATION ==================
#     winner = result_df.iloc[0]['Team']
#     prob = result_df.iloc[0]['Win Probability']

#     st.subheader("🧠 Match Analysis")

#     explanation = f"""
# 🔥 {winner} has higher chances of winning ({prob:.2f}%)

# 📌 Key Factors:
# """

#     if toss_winner == winner:
#         explanation += "\n✔ Won the toss advantage"

#     if toss_decision == 'field' and weather == 'Dew':
#         explanation += "\n✔ Dew favors chasing"

#     if venue in df[df['winner'] == winner]['venue'].values:
#         explanation += "\n✔ Strong venue record"

#     explanation += f"\n🌦️ Weather: {weather}"

#     st.info(explanation)







import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="IPL Live Predictor", layout="centered")

# 🔥 REMOVE WHITE BACKGROUND GLOBALLY
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'

st.title("🏏 WinPredict IPL : IPL Match Winner Prediction")

# ================= UI =================
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

# ================= LOAD =================
model = pickle.load(open("live_model.pkl","rb"))
cols = pickle.load(open("columns.pkl","rb"))
df = pd.read_csv("matches.csv")

df['venue'] = df['venue'].str.strip()
venues = sorted(df['venue'].dropna().unique())

teams = [
    'Chennai Super Kings','Mumbai Indians','Royal Challengers Bangalore',
    'Kolkata Knight Riders','Delhi Capitals','Rajasthan Royals',
    'Sunrisers Hyderabad','Punjab Kings','Gujarat Titans'
]

# ================= INPUT =================
batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team])
venue = st.selectbox("🏟️ Match Venue", venues)

target = st.number_input("Target", 50, 300, 160)
score = st.number_input("Current Score", 0, 300, 80)
overs = st.number_input("Overs", 0.0, 20.0, 10.0, step=0.1)
wickets = st.number_input("Wickets", 0, 10, 2)

# ================= CALCULATIONS =================
runs_left = target - score
balls_left = 120 - int(overs * 6)
wickets_left = 10 - wickets

crr = score / overs if overs > 0 else 0
rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

# ================= PREDICTION =================
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

    input_dict[f'batting_team_{batting_team}'] = 1
    input_dict[f'bowling_team_{bowling_team}'] = 1

    input_df = pd.DataFrame([input_dict])[cols]
    proba = model.predict_proba(input_df)[0]

    win = round(proba[1]*100, 2)
    lose = round(proba[0]*100, 2)

    # ================= RESULT =================
    st.subheader("🎯 Match Insight")

    if win > lose:
        st.success(f"🔥 {batting_team} leads with **{win}%** chance")
        st.progress(int(win))
    else:
        st.error(f"⚡ {bowling_team} leads with **{lose}%** chance")
        st.progress(int(lose))

    # ================= WIN PROBABILITY =================
    st.subheader("📊 Win Probability")

    fig1, ax1 = plt.subplots(figsize=(6,4))
    fig1.patch.set_alpha(0)
    ax1.set_facecolor('none')

    ax1.bar([batting_team, bowling_team], [win, lose],
            color=['#e74c3c', '#3498db'])

    plt.xticks(rotation=10)
    st.pyplot(fig1)

    # ================= HEAD TO HEAD =================
    st.subheader("🥧 Head-to-Head")

    h2h_df = df[
        ((df['team1'] == batting_team) & (df['team2'] == bowling_team)) |
        ((df['team1'] == bowling_team) & (df['team2'] == batting_team))
    ]

    if h2h_df.shape[0] > 0:
        total_matches = h2h_df.shape[0]

        h2h_wins = h2h_df['winner'].value_counts()
        h2h_wins = h2h_wins.reindex([batting_team, bowling_team], fill_value=0)

        st.markdown(f"""
        <div style="text-align:center; font-size:16px;">
        Total Matches: <b>{total_matches}</b><br>
        {batting_team} Wins: <b>{h2h_wins[batting_team]}</b><br>
        {bowling_team} Wins: <b>{h2h_wins[bowling_team]}</b>
        </div>
        """, unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(6,4))
        fig2.patch.set_alpha(0)
        ax2.set_facecolor('none')

        values = [v if v > 0 else 0.1 for v in h2h_wins.values]

        ax2.pie(
            values,
            labels=h2h_wins.index,
            autopct='%1.1f%%',
            colors=['#f39c12','#2ecc71'],
            startangle=90,
            radius=1.2
        )

        ax2.set_aspect('equal')
        st.pyplot(fig2)
    else:
        st.warning("No head-to-head matches found.")

    # ================= MATCH OVERVIEW =================
    st.subheader("📊 Match Overview")

    fig4, ax4 = plt.subplots(figsize=(6,4))
    fig4.patch.set_alpha(0)
    ax4.set_facecolor('none')

    # 🔥 COLORFUL BARS
    colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#5f27cd', '#1dd1a1']

    ax4.bar(
        ["Runs Left","Balls Left","Wickets","CRR","RRR"],
        [runs_left, balls_left, wickets_left, crr, rrr],
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

    # ================= VENUE =================
    st.subheader("🏟️ Venue Head-to-Head")

    venue_df = df[
        (df['venue'] == venue) &
        (
            ((df['team1'] == batting_team) & (df['team2'] == bowling_team)) |
            ((df['team1'] == bowling_team) & (df['team2'] == batting_team))
        )
    ]

    if venue_df.shape[0] > 0:
        total_matches_v = venue_df.shape[0]

        venue_wins = venue_df['winner'].value_counts()
        venue_wins = venue_wins.reindex([batting_team, bowling_team], fill_value=0)

        st.markdown(f"""
        <div style="text-align:center; font-size:16px;">
        Total Matches: <b>{total_matches_v}</b><br>
        {batting_team} Wins: <b>{venue_wins[batting_team]}</b><br>
        {bowling_team} Wins: <b>{venue_wins[bowling_team]}</b>
        </div>
        """, unsafe_allow_html=True)

        fig5, ax5 = plt.subplots(figsize=(6,4))
        fig5.patch.set_alpha(0)
        ax5.set_facecolor('none')

        ax5.bar(venue_wins.index, venue_wins.values,
                color=['#c0392b','#27ae60'])

        plt.xticks(rotation=10)
        st.pyplot(fig5)
    else:
        st.warning("No matches between these teams at this venue.")