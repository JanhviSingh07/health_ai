import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("data.csv")

# Encode target
le = LabelEncoder()
df['stress_level'] = le.fit_transform(df['stress_level'])

# Features
X = df[['sleep','exercise','screen_time','diet_quality','bmi']]
y = df['stress_level']

# 🧠 1. Classification Model
clf = LogisticRegression()
clf.fit(X, y)

# 🧠 2. Regression Model (Health Score)
df['health_score'] = 100 - (df['bmi']*2 + df['screen_time']*3 - df['exercise']*5)
y_reg = df['health_score']

reg = RandomForestRegressor()
reg.fit(X, y_reg)

# 🧠 3. Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)


# 🎯 PREDICT FUNCTION
def predict_all(input_data):
    df_input = pd.DataFrame([input_data])

    risk = clf.predict(df_input)[0]
    score = reg.predict(df_input)[0]
    cluster = kmeans.predict(df_input)[0]

    return risk, score, cluster