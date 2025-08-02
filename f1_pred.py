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
    'driver': ['Charles Leclerc','Oscar Piastri','Lando Norris','George Russell','Fernando Alonso',
               'Lance Stroll','Gabriel Bortoleto','Max Verstappen','Liam Lawson','Isack Hadjar',
               'Oliver Bearman','Lewis Hamilton','Carlos Sainz','Franco Colapinto','Kimi Antonelli',
               'Yuki Tsunoda','Pierre Gasly','Esteban Ocon','Nico Hulkenberg','Alexander Albon'],
    'driver_points':[139,266,250,157,16,20,6,185,16,22,
                     8,109,16,0,63,10,20,27,37,54],
    'qualifying_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'constructor': ['Ferrari','McLaren','McLaren','Mercedes','Aston Martin',
                    'Aston Martin','Kick Sauber','Red Bull','Racing Bulls','Racing Bulls',
                    'Haas','Ferrari','Williams','Alpine','Mercedes',
                    'Red Bull','Alpine','Haas','Kick Sauber','Williams'],
    'constructor_points':[248,516,516,220,36,36,43,192,41,41,
                          35,248,70,20,220,192,20,35,43,70],
    'fastest_lap_s': [75.372,75.398,75.413,75.425,75.481,75.498,75.725,75.728,75.821,75.915,
                      75.694,75.702,75.781,76.159,76.386,75.899,75.966,76.023,76.081,76.223],
    'average_speed': [209.2,209.1,209.1,209.1,208.9,208.9,208.2,208.2,208.0,207.7,
                      208.3,208.3,208.1,207.0,206.4,207.7,207.6,207.4,207.3,206.9],
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