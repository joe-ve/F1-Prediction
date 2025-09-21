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
    'driver': ['Max Verstappen','Carlos Sainz','Liam Lawson','Kimi Antonelli','George Russell',
               'Yuki Tsunoda','Lando Norris','Isack Hadjar','Oscar Piastri','Charles Leclerc',
               'Fernando Alonso','Lewis Hamilton','Gabriel Bortoleto','Lance Stroll','Oliver Bearman',
               'Franco Colapinto','Nico Hulkenberg','Pierre Gasly','Alexander Albon','Esteban Ocon'],
    'driver_points':[230,16,20,66,194,12,293,38,324,163,
                     30,117,18,32,16,0,37,20,70,28],
    'qualifying_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'constructor': ['Red Bull','Williams','Racing Bulls','Mercedes','Mercedes',
                    'Red Bull','McLaren','Racing Bulls','McLaren','Ferrari',
                    'Aston Martin','Ferrari','Kick Sauber','Aston Martin','Haas',
                    'Alpine','Kick Sauber','Alpine','Williams','Haas'],
    'constructor_points':[239,86,61,260,260,239,617,61,617,280,
                          62,280,55,62,44,20,55,20,86,44],
    'fastest_lap_s': [101.117,101.595,101.707,101.717,102.070,102.143,102.239,102.372,101.414,101.519,
                      101.857,102.183,102.277,103.061,102.666,102.779,102.916,103.139,103.778,0],
    'average_speed': [213.7,212.7,212.4,212.4,211.7,211.5,211.3,211.1,0,0,
                      212.1,211.4,211.2,209.6,0,210.2,209.9,209.5,208.2,0],
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