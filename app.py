import streamlit as st
import pandas as pd
from model import predict_all, get_feature_importance

st.set_page_config(page_title="Health AI", layout="wide")

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.score {
    font-size:60px !important;
    font-weight: bold;
    color: #00ffcc;
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
# INPUTS
# =========================
col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("😴 Sleep Hours (per day)", 0, 10)
    exercise = st.slider("🏃 Exercise (hours/day)", 0, 5)
    diet = st.slider("🥗 Diet Quality (1-5)", 1, 5)

with col2:
    screen = st.slider("📱 Screen Time (hours/day)", 0, 12)
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

    risk, score, cluster, reasons, risk_classes = predict_all(input_data)

    cluster_labels = ["Sedentary", "Balanced", "Active"]

    # =========================
    # RESULTS
    # =========================
    st.markdown("## 📊 Your Health Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='score'>{round(score,1)}</div>", unsafe_allow_html=True)
        st.write("Health Score")

    with col2:
        st.success(f"⚠ Stress Level: {risk_classes[risk]}")

    with col3:
        st.info(f"🧬 Lifestyle: {cluster_labels[cluster]}")

    st.divider()

    # =========================
    # WHY
    # =========================
    st.subheader("🧠 Why this result?")
    for r in reasons:
        st.write(f"👉 {r}")

    # =========================
    # FEATURE IMPORTANCE 🔥
    # =========================
    st.subheader("📊 What affects stress most?")

    importance = get_feature_importance()

    imp_df = pd.DataFrame({
        "Feature": list(importance.keys()),
        "Impact": list(importance.values())
    })

    st.bar_chart(imp_df.set_index("Feature"))

    # =========================
    # FUTURE SIMULATION
    # =========================
    st.divider()
    st.subheader("🔮 Future Simulation")

    improved_sleep = min(sleep + 2, 10)
    improved_exercise = min(exercise + 1, 5)

    new_input = {
        'sleep': improved_sleep,
        'exercise': improved_exercise,
        'screen_time': screen,
        'diet_quality': diet,
        'bmi': bmi
    }

    _, new_score, _, _, _ = predict_all(new_input)

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