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
    'driver': ['Lando Norris','Max Verstappen','Carlos Sainz','George Russell','Oscar Piastri',
               'Liam Lawson','Fernando Alonso','Isack Hadjar','Charles Leclerc','Pierre Gasly',
               'Nico Hulkenberg','Lance Stroll','Esteban Ocon','Oliver Bearman','Franco Colapinto',
               'Alexander Albon','Kimi Antonelli','Gabriel Bortoleto','Yuki Tsunoda','Lewis Hamilton'],
    'driver_points':[390,341,38,276,366,36,40,43,214,22,
                     43,32,30,40,0,73,122,19,28,148],
    'qualifying_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'constructor': ['McLaren','Red Bull','Williams','Mercedes','McLaren',
                    'Racing Bulls','Aston Martin','Racing Bulls','Ferrari','Alpine',
                    'Kick Sauber','Aston Martin','Haas','Haas','Alpine',
                    'Williams','Mercedes','Kick Sauber','Red Bull','Ferrari'],
    'constructor_points':[756,366,111,398,756,82,72,82,362,22,
                          62,72,70,70,22,111,398,62,366,362],
    'fastest_lap_s': [107.934,108.257,108.296,108.803,108.961,109.062,109.466,109.554,109.872,111.540,
                      112.781,112.850,112.987,113.094,113.683,116.220,116.314,116.674,116.798,117.115],
    'average_speed': [206.8,206.2,206.1,205.1,204.8,204.6,203.9,203.7,203.1,200.1,
                      197.9,197.8,197.5,197.3,196.3,192.0,191.9,191.3,191.1,190.6],
    'pit_stops': [2]*num,
    'dnf': ['No']*num,
    'circuit_type': ['Street']*num,
    'weather': ['Wet']*num
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