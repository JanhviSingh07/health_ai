import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
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
# HANDLE BMI
# =========================
bmi_map = {
    'Underweight': 18,
    'Normal': 22,
    'Normal Weight': 22,
    'Overweight': 27,
    'Obese': 32
}
df['bmi'] = df['bmi_category'].map(bmi_map)

# =========================
# FEATURE ENGINEERING
# =========================
np.random.seed(42)
df['screen_time'] = (10 - df['sleep'] + np.random.randint(0, 3, len(df))).clip(1, 12)

df.dropna(inplace=True)

# =========================
# ENCODE TARGET
# =========================
le = LabelEncoder()
df['stress_level'] = le.fit_transform(df['stress_level'])
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
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# MODEL 1: CLASSIFICATION
# =========================
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = clf.predict(X_test)
print("🔥 Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# FEATURE IMPORTANCE 🔥
# =========================
feature_importance = dict(zip(features, clf.feature_importances_))

def get_feature_importance():
    return feature_importance

# =========================
# HEALTH SCORE (RULE-BASED)
# =========================
def calculate_health_score(row):
    score = 100

    if row['sleep'] < 5:
        score -= 30
    elif row['sleep'] < 7:
        score -= 10
    elif row['sleep'] > 9:
        score -= 5

    if row['screen_time'] > 8:
        score -= 25
    elif row['screen_time'] > 5:
        score -= 10

    if row['exercise'] == 0:
        score -= 25
    elif row['exercise'] < 2:
        score -= 10
    elif row['exercise'] > 4:
        score += 5

    if row['diet_quality'] <= 2:
        score -= 20
    elif row['diet_quality'] >= 4:
        score += 5

    if row['bmi'] > 30:
        score -= 20
    elif row['bmi'] > 25:
        score -= 10

    return max(0, min(score, 100))

# =========================
# CLUSTERING
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# =========================
# EXPLAINABLE AI
# =========================
def get_reasons(input_data):
    reasons = []

    if input_data['sleep'] < 6:
        reasons.append("Low sleep")
    if input_data['screen_time'] > 7:
        reasons.append("High screen time")
    if input_data['exercise'] < 2:
        reasons.append("Low physical activity")
    if input_data['diet_quality'] <= 2:
        reasons.append("Poor diet")
    if input_data['bmi'] > 27:
        reasons.append("High BMI")

    if not reasons:
        reasons.append("Healthy lifestyle")

    return reasons

# =========================
# FINAL PREDICT FUNCTION (HYBRID AI 🔥)
# =========================
def predict_all(input_data):

    df_input = pd.DataFrame([input_data])
    df_input = df_input[['sleep','exercise','screen_time','diet_quality','bmi']]

    input_scaled = scaler.transform(df_input)

    # ML prediction
    risk_ml = clf.predict(input_scaled)[0]

    # HYBRID RULE FIX
    sleep = input_data['sleep']
    screen = input_data['screen_time']
    exercise = input_data['exercise']
    diet = input_data['diet_quality']

    if sleep < 5 and screen > 8:
        risk = 2  # High
    elif sleep < 6 or exercise < 1 or diet <= 2:
        risk = max(risk_ml, 1)
    else:
        risk = risk_ml

    # Rule-based score
    score = calculate_health_score(df_input.iloc[0])

    cluster = kmeans.predict(input_scaled)[0]
    reasons = get_reasons(input_data)

    return int(risk), float(score), int(cluster), reasons, risk_classes