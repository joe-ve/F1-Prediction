import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_excel("f1/F1.xlsx", sheet_name='2025')

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
    'driver': ['Lando Norris','Charles Leclerc','Oscar Piastri','Lewis Hamilton','Max Verstappen',
               'Isack Hadjar','Fernando Alonso','Esteban Ocon','Liam Lawson','Alexander Albon',
               'Carlos Sainz','Yuki Tsunoda','Nico Hulkenberg','George Russell','Kimi Antonelli',
               'Gabriel Bortoleto','Oliver Bearman','Pierre Gasly','Lance Stroll','Franco Colapinto'],
    'driver_points':[133,61,146,53,124,7,0,14,0,40,
                     11,10,6,99,48,0,6,7,14,0],
    'qualifying_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'constructor': ['McLaren','Ferrari','McLaren','Ferrari','Red Bull',
                    'Racing Bulls','Aston Martin','Haas','Racing Bulls','Williams',
                    'Williams','Red Bull','Kick Sauber','Mercedes','Mercedes',
                    'Kick Sauber','Haas','Alpine','Aston Martin','Alpine'],
    'constructor_points':[279,114,279,114,131,10,14,20,10,51,
                          51,131,6,147,147,6,20,7,14,7],
    'fastest_lap_s': [69.954,70.063,70.129,70.382,70.669,70.923,70.924,70.942,71.129,71.213,
                      71.362,71.415,71.596,71.507,71.880,71.902,71.979,71.994,72.563,72.597],
    'average_speed': [171.7,171.4,171.3,170.6,169.9,169.3,169.3,169.3,168.8,168.6,
                      168.3,168.2,167.7,167.5,167.2,167.0,166.8,166.8,165.5,165.4],
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