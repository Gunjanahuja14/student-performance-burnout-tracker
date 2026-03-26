import pandas as pd
import random

data = []

for _ in range(300):
    study = random.randint(1, 12)
    sleep = random.randint(3, 9)
    stress = random.randint(1, 10)
    screen = random.randint(1, 10)
    fatigue = random.randint(1, 10)

    burnout_score = (0.4 * stress + 
                     0.3 * fatigue + 
                     0.2 * screen + 
                     0.1 * (10 - sleep))
    
    burnout = 1 if burnout_score > 5 else 0

    data.append([study, sleep, stress, screen, fatigue, burnout])

df = pd.DataFrame(data, columns=[
    "study_hours", 
    "sleep_hours", 
    "stress_level", 
    "screen_time", 
    "mental_fatigue", 
    "burnout"
])

df.to_csv("data/student_data.csv", index=False)

print("Dataset created in data folder!")
