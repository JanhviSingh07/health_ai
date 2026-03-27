import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# =========================
# CLEAN COLUMN NAMES
# =========================
df.columns = df.columns.str.strip()

# =========================
# RENAME COLUMNS
# =========================
df.rename(columns={
    'Sleep Duration': 'sleep',
    'Physical Activity Level': 'exercise',
    'Quality of Sleep': 'diet_quality',
    'Stress Level': 'stress_level',
    'BMI Category': 'bmi_category'
}, inplace=True)

# =========================
# HANDLE BMI CATEGORY
# =========================
bmi_map = {
    'Underweight': 18,
    'Normal': 22,
    'Overweight': 27,
    'Obese': 32,
    'Normal Weight': 22
}

df['bmi'] = df['bmi_category'].map(bmi_map)

# =========================
# FEATURE ENGINEERING
# =========================
df['screen_time'] = (
    10 - df['sleep'] + np.random.randint(0, 3, size=len(df))
).clip(1, 12)

# =========================
# CLEAN DATA
# =========================
df.dropna(inplace=True)

# =========================
# ENCODE TARGET
# =========================
le = LabelEncoder()
df['stress_level'] = le.fit_transform(df['stress_level'])

# Save readable labels
risk_classes = list(le.classes_)

# =========================
# FEATURES
# =========================
features = ['sleep','exercise','screen_time','diet_quality','bmi']
X = df[features]
y = df['stress_level']

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# MODEL 1: CLASSIFICATION
# =========================
clf = LogisticRegression(max_iter=1000)
clf.fit(X_scaled, y)

# =========================
# MODEL 2: HEALTH SCORE (RULE BASED)
# =========================
def calculate_health_score(input_data):

    score = 100

    sleep = input_data['sleep']
    screen = input_data['screen_time']
    exercise = input_data['exercise']
    diet = input_data['diet_quality']
    bmi = input_data['bmi']

    # Sleep
    if sleep < 5:
        score -= 30
    elif sleep < 7:
        score -= 10
    elif sleep > 9:
        score -= 5

    # Screen time
    if screen > 9:
        score -= 25
    elif screen > 6:
        score -= 10

    # Exercise
    if exercise == 0:
        score -= 25
    elif exercise < 2:
        score -= 10
    elif exercise > 5:
        score += 5

    # Diet
    if diet <= 2:
        score -= 20
    elif diet >= 4:
        score += 5

    # BMI
    if bmi > 30:
        score -= 20
    elif bmi > 25:
        score -= 10

    return max(0, min(score, 100))

# =========================
# MODEL 3: CLUSTERING
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# =========================
# PREDICT FUNCTION
# =========================
def predict_all(input_data):

    df_input = pd.DataFrame([input_data])

    # Ensure correct order
    df_input = df_input[['sleep','exercise','screen_time','diet_quality','bmi']]

    # Scale input
    input_scaled = scaler.transform(df_input)

    # Predictions
    risk = clf.predict(input_scaled)[0]
    cluster = kmeans.predict(input_scaled)[0]

    # Rule-based score (IMPORTANT FIX)
    score = calculate_health_score(input_data)

    return int(risk), float(score), int(cluster), risk_classes