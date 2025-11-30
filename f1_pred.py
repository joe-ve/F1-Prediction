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
    'driver': ['Oscar Piastri','Lando Norris','Max Verstappen','George Russell','Kimi Antonelli',
               'Isack Hadjar','Carlos Sainz','Fernando Alonso','Pierre Gasly','Charles Leclerc',
               'Nico Hulkenberg','Liam Lawson','Oliver Bearman','Gabriel Bortoleto','Alexander Albon',
               'Yuki Tsunoda','Esteban Ocon','Lewis Hamilton','Lance Stroll','Franco Colapinto'],
    'driver_points':[374,396,371,301,140,51,49,42,22,226,
                     49,36,41,19,73,32,32,152,32,0],
    'qualifying_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'constructor': ['McLaren','McLaren','Red Bull','Mercedes','Mercedes',
                    'Racing Bulls','Williams','Aston Martin','Alpine','Ferrari',
                    'Kick Sauber','Racing Bulls','Haas','Kick Sauber','Williams',
                    'Red Bull','Haas','Ferrari','Aston Martin','Alpine'],
    'constructor_points':[770,770,400,441,441,90,122,74,22,378,
                          68,90,73,68,122,400,73,378,74,22],
    'fastest_lap_s': [79.387,79.495,79.651,79.662,79.846,80.114,80.287,80.418,80.477,80.561,
                      80.353,80.433,80.438,80.534,80.629,80.761,80.864,80.907,81.058,81.137],
    'average_speed': [245.7,245.4,244.9,244.8,244.3,243.5,242.9,242.5,242.4,242.1,
                      242.7,242.5,242.5,242.2,241.9,241.5,241.2,241.1,240.6,240.4],
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