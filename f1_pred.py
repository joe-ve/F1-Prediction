import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_excel("F1.xlsx", sheet_name='2025')

df = df.dropna(subset=['Fastest_Lap(m)','Fastest_Lap(s)','Average_Speed','Pit_Stops'])

df['Circuit_Code'] = df['Circuit_Type'].astype('category').cat.codes
df['Weather_Code'] = df['Weather'].astype('category').cat.codes
df['Driver_Code'] = df['Driver_Name'].astype('category').cat.codes
df['Team_Code'] = df['Team_Name'].astype('category').cat.codes
df['DNF_Flag'] = df['DNF'].map({'Yes':1, 'No':0})

X = df[['Circuit_Code','Weather_Code','Driver_Code','Driver_Points','Team_Code','Team_Points','Qualifying_Position',
       'Fastest_Lap(s)','Average_Speed','Pit_Stops','DNF_Flag']]
Y = df['Winner']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, Y_pred))

circuit_map = dict(zip(df['Circuit_Type'], df['Circuit_Code']))
weather_map = dict(zip(df['Weather'], df['Weather_Code']))
driver_map = dict(zip(df['Driver_Name'], df['Driver_Code']))
team_map = dict(zip(df['Team_Name'], df['Team_Code']))

num = 20
upcoming_race = pd.DataFrame({
    'driver': ['Lando Norris','Oscar Piastri','Charles Leclerc','Max Verstappen','Alexander Albon',
               'George Russell','Yuki Tsunoda','Isack Hadjar','Liam Lawson','Gabriel Bortoleto',
               'Esteban Ocon','Oliver Bearman','Pierre Gasly','Nico Hulkenberg','Carlos Sainz',
               'Lewis Hamilton','Franco Colapinto','Kimi Antonelli','Fernando Alonso','Lance Stroll'],
    'driver_points':[232,241,124,173,46,147,10,22,12,4,
                     27,8,19,37,16,103,0,63,16,20],
    'qualifying_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'constructor': ['McLaren','McLaren','Ferrari','Red Bull','Williams',
                    'Mercedes','Red Bull','Racing Bulls','Racing Bulls','Kick Sauber',
                    'Haas','Haas','Alpine','Kick Sauber','Williams',
                    'Ferrari','Alpine','Mercedes','Aston Martin','Aston Martin'],
    'constructor_points':[473,473,227,180,62,210,180,37,37,41,
                          35,35,19,41,62,227,19,210,36,36],
    'fastest_lap_s': [100.562,100.647,100.900,100.903,101.201,101.260,101.284,101.310,101.328,102.387,
                      101.525,101.617,101.633,101.707,101.758,101.939,102.022,102.139,102.385,102.502],
    'average_speed': [250.7,250.5,249.8,249.8,249.1,249.0,248.9,248.8,248.8,246.2,
                      248.3,248.1,248.0,247.9,247.7,247.3,247.1,246.8,246.2,245.9],
    'pit_stops': [2]*num,
    'dnf': ['No']*num,
    'circuit_type': ['Permanent']*num,
    'weather': ['Dry']*num
})

upcoming_race['Circuit_Code'] = upcoming_race['circuit_type'].map(circuit_map)
upcoming_race['Weather_Code'] = upcoming_race['weather'].map(weather_map)
upcoming_race['Driver_Code'] = upcoming_race['driver'].map(driver_map)
upcoming_race['Team_Code'] = upcoming_race['constructor'].map(team_map)

upcoming_race['Driver_Points'] = upcoming_race['driver_points']
upcoming_race['Team_Points'] = upcoming_race['constructor_points']
upcoming_race['Qualifying_Position'] = upcoming_race['qualifying_position']
upcoming_race['Fastest_Lap(s)'] = upcoming_race['fastest_lap_s']
upcoming_race['Average_Speed'] = upcoming_race['average_speed']
upcoming_race['Pit_Stops'] = upcoming_race['pit_stops']
upcoming_race['DNF_Flag'] = upcoming_race['dnf'].map({'Yes': 1, 'No': 0})

X_upcoming = upcoming_race[[
    'Circuit_Code', 'Weather_Code', 'Driver_Code','Driver_Points', 'Team_Code',
    'Team_Points','Qualifying_Position', 'Fastest_Lap(s)',
    'Average_Speed', 'Pit_Stops', 'DNF_Flag'
]]

win_probs = model.predict_proba(X_upcoming)[:, 1]
upcoming_race['Win_Probability'] = win_probs

top_3_preds = upcoming_race.sort_values(by='Win_Probability', ascending=False).head(3)
print("\n🏁 Top 3 Predicted Winners for Upcoming GP:")
print(top_3_preds[['driver', 'constructor', 'Win_Probability']])