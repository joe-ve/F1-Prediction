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
    'driver': ['George Russell','Max Verstappen','Oscar Piastri','Kimi Antonelli','Lewis Hamilton',
               'Fernando Alonso','Lando Norris','Charles Leclerce','Isack Hadjar','Alexander Albon',
               'Yuki Tsunoda','Franco Colapinto','Nico Hulkenberg','Oliver Bearman','Esteban Ocon',
               'Gabriel Bortoleto','Carlos Sainz','Lance Stroll','Liam Lawson','Pierre Gasly'],
    'driver_points':[111,137,186,48,71,2,176,94,21,42,
                     10,0,16,6,20,0,12,14,4,11],
    'qualifying_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'constructor': ['Mercedes','Red Bull','McLaren','Mercedes','Ferrari',
                    'Aston Martin','McLaren','Ferrari','Racing Bulls','Williams',
                    'Red Bull','Alpine','Kick Sauber','Haas','Haas',
                    'Kick Sauber','Williams','Aston Martin','Racing Bulls','Alpine'],
    'constructor_points':[159,144,362,159,165,16,362,165,28,54,
                          144,11,16,26,26,16,54,16,28,11],
    'fastest_lap_s': [70.899,71.059,71.120,71.391,71.526,71.586,71.625,71.682,71.867,71.907,
                      72.102,72.142,72.183,72.340,72.634,72.385,72.398,72.517,72.525,72.667],
    'average_speed': [221.4,220.9,220.7,219.9,219.4,219.3,219.1,219.0,218.4,218.3,
                      217.7,217.6,217.4,217.0,216.1,216.8,216.8,216.4,216.4,216.0],
    'pit_stops': [1]*num,
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