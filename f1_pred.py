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
    'driver': ['Oscar Piastri','Max Verstappen','George Russell','Lando Norris','Fernando Alonso',
               'Carlos Sainz','Alexander Albon','Lance Stroll','Isack Hadjar','Pierre Gasly',
               'Charles Leclerc','Lewis Hamilton','Kimi Antonelli','Gabriel Bortoleto','Franco Colapinto',
               'Liam Lawson','Nico Hulkenberg','Esteban Ocon','Oliver Bearman','Yuki Tsunoda'],
    'driver_points':[131,99,93,115,0,7,30,14,5,7,
                     53,41,48,0,0,0,6,14,6,9],
    'qualifying_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'constructor': ['McLaren','Red Bull','Mercedes','McLaren','Aston Martin',
                    'Williams','Williams','Aston Martin','Racing Bulls','Alpine',
                    'Ferrari','Ferrari','Mercedes','Kick Sauber','Alpine',
                    'Racing Bulls','Kick Sauber','Haas','Haas','Red Bull'],
    'constructor_points':[246,105,141,246,14,37,37,14,8,7,
                          94,94,141,6,7,8,6,20,20,105],
    'fastest_lap_s': [74.670,74.704,74.807,74.962,75.431,75.432,75.473,75.581,75.746,75.787,
                      75.604,75.765,75.772,76.260,76.256,76.379,76.518,76.613,76.918,77.124],
    'average_speed': [236.6,236.5,236.2,235.7,234.2,234.2,234.1,233.8,233.3,233.1,
                      233.7,233.2,233.2,231.7,231.5,231.3,230.9,230.6,229.7,229.4],
    'pit_stops': [2]*num,
    'dnf': ['No']*num,
    'circuit_type': ['Street']*num,
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