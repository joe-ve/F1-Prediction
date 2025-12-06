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
    'driver': ['Max Verstappen','Lando Norris','Oscar Piastri','George Russell','Charles Leclerc',
               'Fernando Alonso','Gabriel Bortoleto','Esteban Ocon','Isack Hadjar','Yuki Tsunoda',
               'Oliver Bearman','Carlos Sainz','Liam Lawson','Kimi Antonelli','Lance Stroll',
               'Lewis Hamilton','Alexander Albon','Nico Hulkenberg','Pierre Gasly','Franco Colapinto'],
    'driver_points':[396,408,392,309,230,48,19,32,51,33,
                     41,64,38,150,32,152,73,49,22,0],
    'qualifying_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'constructor': ['Red Bull','McLaren','McLaren','Mercedes','Ferrari',
                    'Aston Martin','Kick Sauber','Haas','Racing Bulls','Red Bull',
                    'Haas','Williams','Racing Bulls','Mercedes','Aston Martin',
                    'Ferrari','Williams','Kick Sauber','Alpine','Alpine'],
    'constructor_points':[426,800,800,459,382,80,68,73,92,426,
                          73,137,92,459,80,382,137,68,22,22],
    'fastest_lap_s': [82.207,82.408,82.437,82.645,82.730,82.902,82.904,82.913,83.072,83.034,
                      83.041,83.042,83.077,83.080,83.097,83.394,83.416,83.450,83.468,83.890],
    'average_speed': [231.2,230.7,230.6,230.0,229.8,229.3,229.3,229.2,228.8,228.8,
                      228.9,228.9,228.8,228.8,228.7,227.9,227.9,227.8,227.7,226.6],
    'pit_stops': [1]*num,
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