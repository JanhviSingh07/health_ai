import streamlit as st
import pandas as pd
from model import predict_all

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Health AI", layout="wide")

# =========================
# CUSTOM CSS ( REAL UI BOOST)
# =========================
st.markdown("""
<style>
.big-font {
    font-size:28px !important;
    font-weight: bold;
}
.score {
    font-size:60px !important;
    font-weight: bold;
    color: #00ffcc;
}
.card {
    padding:20px;
    border-radius:15px;
    background-color:#111;
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("🧠 AI Health Digital Twin")
st.markdown("### 🚀 Predict your health, understand risks & simulate improvements")

st.divider()

# =========================
# INPUTS (2 COLUMN LAYOUT )
# =========================
col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("😴 Sleep Hours", 0, 10)
    exercise = st.slider("🏃 Exercise (hours/week)", 0, 10)
    diet = st.slider("🥗 Diet Quality (1-5)", 1, 5)

with col2:
    screen = st.slider("📱 Screen Time", 0, 12)
    bmi = st.slider("⚖ BMI", 15, 35)

st.divider()

# =========================
# BUTTON
# =========================
if st.button("🚀 Analyze Now"):

    input_data = {
        'sleep': sleep,
        'exercise': exercise,
        'screen_time': screen,
        'diet_quality': diet,
        'bmi': bmi
    }

    risk, score, cluster = predict_all(input_data)

    risk_labels = ["Low", "Medium", "High"]
    cluster_labels = ["Sedentary", "Balanced", "Active"]

    # =========================
    #  HERO SECTION (BIG SCORE)
    # =========================
    st.markdown("## 📊 Your Health Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='score'>{round(score,1)}</div>", unsafe_allow_html=True)
        st.write("Health Score")

    with col2:
        if risk == 2:
            st.error(f"⚠ {risk_labels[risk]}")
        elif risk == 1:
            st.warning(f"⚠ {risk_labels[risk]}")
        else:
            st.success(f"✅ {risk_labels[risk]}")

    with col3:
        st.info(f"🧬 {cluster_labels[cluster]}")

    st.divider()

    # =========================
    # EXPLANATION
    # =========================
    st.subheader("🧠 Why this result?")

    reasons = []

    if sleep < 6:
        reasons.append("Low sleep is increasing your risk")
    if screen > 7:
        reasons.append("High screen time affecting health")
    if exercise < 2:
        reasons.append("Lack of physical activity")
    if diet <= 2:
        reasons.append("Poor diet quality")
    if bmi > 27:
        reasons.append("High BMI contributing to risk")

    if reasons:
        for r in reasons:
            st.write(f"👉 {r}")
    else:
        st.success("Great lifestyle balance!")

    # =========================
    # FUTURE SIMULATION
    # =========================
    st.divider()
    st.subheader("🔮 Future Simulation")

    improved_sleep = min(sleep + 2, 10)
    improved_exercise = min(exercise + 1, 10)

    new_input = {
        'sleep': improved_sleep,
        'exercise': improved_exercise,
        'screen_time': screen,
        'diet_quality': diet,
        'bmi': bmi
    }

    _, new_score, _ = predict_all(new_input)

    improvement = new_score - score

    if improvement > 0:
        st.success(f"🚀 +{round(improvement,2)} improvement expected")
    else:
        st.error("⚠ No improvement")

    # =========================
    # GRAPH
    # =========================
    st.subheader("📊 Comparison")

    chart_data = pd.DataFrame({
        "Type": ["Current", "Future"],
        "Score": [score, new_score]
    })

    st.bar_chart(chart_data.set_index("Type"))