import pandas as pd
import random

random.seed(42)

data = []

for _ in range(500):

    study   = random.randint(1, 12)
    sleep   = random.randint(3, 9)
    stress  = random.randint(1, 10)
    screen  = random.randint(1, 10)
    fatigue = random.randint(1, 10)

    score = 0

    # ── Linear contributions ──────────────────────────────────────────────
    if stress >= 7:
        score += 3

    if fatigue >= 7:
        score += 2

    if sleep <= 5:
        score += 2

    if screen >= 7:
        score += 1

    if study >= 9:
        score += 1

    # ── Nonlinear interaction patterns ────────────────────────────────────
    # These create sharp decision boundaries that a tree captures perfectly
    # but logistic regression (a linear model) cannot fit without feature
    # engineering, so the two models end up with meaningfully different accuracy.

    if stress >= 7 and sleep <= 5:       # exhausted + high stress → strong signal
        score += 4

    if fatigue >= 8 and screen >= 7:     # mental fatigue compounded by screen time
        score += 3

    if stress >= 8 and fatigue >= 8:     # dual overload
        score += 3

    if study >= 9 and sleep <= 4:        # over-studying on little sleep
        score += 2

    if sleep <= 4 and screen >= 8:       # poor sleep with high screen time
        score += 2

    # ── Final classification ──────────────────────────────────────────────
    burnout = 1 if score >= 8 else 0

    # Controlled noise (4 %) — keeps the dataset realistic but stays low
    # enough for the decision tree to reach ~92 % accuracy.
    if random.random() < 0.04:
        burnout = 1 - burnout

    data.append([study, sleep, stress, screen, fatigue, burnout])

df = pd.DataFrame(data, columns=[
    "study_hours",
    "sleep_hours",
    "stress_level",
    "screen_time",
    "mental_fatigue",
    "burnout",
])

df.to_csv("data/student_data.csv", index=False)

print("Dataset created successfully!")
print(f"Total samples : {len(df)}")
print(f"Burnout = 1   : {df['burnout'].sum()}")
print(f"Burnout = 0   : {(df['burnout'] == 0).sum()}")