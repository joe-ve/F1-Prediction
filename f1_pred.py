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
    'driver': ['Lando Norris','Charles Leclerc','Lewis Hamilton','George Russell','Max Verstappen',
               'Kimi Antonelli','Carlos Sainz','Oscar Piastri','Isack Hadjar','Oliver Bearman',
               'Yuki Tsunoda','Esteban Ocon','Nico Hulkenberg','Fernando Alonso','Liam Lawson',
               'Gabriel Bortoleto','Alexander Albon','Pierre Gasly','Lance Stroll','Franco Colapinto'],
    'driver_points':[332,192,142,252,306,89,38,346,39,20,
                     28,28,41,37,30,18,73,20,32,0],
    'qualifying_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'constructor': ['McLaren','Ferrari','Ferrari','Mercedes','Red Bull',
                    'Mercedes','Ferrari','McLaren','Racing Bulls','Haas',
                    'Red Bull','Haas','Kick Sauber','Aston Martin','Racing Bulls',
                    'Kick Sauber','Williams','Alpine','Aston Martin','Alpine'],
    'constructor_points':[678,334,334,341,331,341,334,678,72,48,
                          331,48,59,69,72,59,111,20,69,20],
    'fastest_lap_s': [75.586,75.848,75.938,76.034,76.070,76.118,76.172,76.174,76.252,76.460,
                      76.816,76.837,77.016,77.103,78.072,77.412,77.490,77.546,77.606,77.670],
    'average_speed': [204.9,204.2,204.0,203.7,203.6,203.5,203.4,203.4,203.2,202.6,
                      201.7,201.6,201.1,200.9,198.4,200.1,199.9,199.8,199.6,199.4],
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