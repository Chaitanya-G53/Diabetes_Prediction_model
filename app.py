import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model.pkl")

# --------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------
# CUSTOM CSS — biotech laboratory vibe
# --------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @keyframes spinHelix {
        from { transform: rotateY(0deg); }
        to   { transform: rotateY(360deg); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 8px rgba(0,255,200,0.35); }
        50%      { box-shadow: 0 0 22px rgba(0,255,200,0.75); }
    }
    @keyframes drift {
        0%   { transform: translateY(0px); }
        50%  { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 20%, rgba(0,255,200,0.06), transparent 40%),
            radial-gradient(circle at 85% 80%, rgba(0,180,255,0.08), transparent 45%),
            linear-gradient(135deg, #020c1b 0%, #041e30 45%, #06283d 100%);
    }

    .lab-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 18px;
        margin-bottom: 4px;
    }
    .lab-header .helix-icon {
        animation: spinHelix 6s linear infinite;
        transform-style: preserve-3d;
    }
    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00ffc8, #00b4ff, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
    }
    .sub-title {
        text-align: center;
        color: #9fd8d0;
        font-size: 1.0rem;
        margin-top: 2px;
        margin-bottom: 6px;
        letter-spacing: 0.3px;
    }
    .icon-strip {
        text-align: center;
        font-size: 1.6rem;
        letter-spacing: 14px;
        opacity: 0.85;
        margin-bottom: 1.2rem;
        animation: drift 4s ease-in-out infinite;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(0,255,200,0.18);
        border-radius: 18px;
        padding: 22px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
        animation: pulseGlow 5s ease-in-out infinite;
        margin-bottom: 16px;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 26px;
        text-align: center;
        border: 1px solid rgba(0,255,200,0.25);
        box-shadow: 0 0 30px rgba(0,255,200,0.12);
    }
    .badge-risk {
        display:inline-block;
        padding: 8px 22px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        background: rgba(255, 82, 82, 0.16);
        color: #ff6b6b;
        border: 1px solid rgba(255,82,82,0.4);
    }
    .badge-safe {
        display:inline-block;
        padding: 8px 22px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        background: rgba(0, 255, 200, 0.14);
        color: #00ffc8;
        border: 1px solid rgba(0,255,200,0.4);
    }
    .care-item {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #00b4ff;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.93rem;
        color: #d7f7f0;
    }
    .care-item.warn {
        border-left-color: #ff6b6b;
    }
    .care-item.good {
        border-left-color: #00ffc8;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #01111d, #04283e);
        border-right: 1px solid rgba(0,255,200,0.15);
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00ffc8, #00b4ff);
        color: #012;
        font-weight: 800;
        border-radius: 10px;
        border: none;
        padding: 0.65em 1.4em;
        width: 100%;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(0,255,200,0.55);
    }
    h1, h2, h3, h4 { color: #e3fdf9; }
    .disclaimer {
        font-size: 0.78rem;
        color: #7d9a94;
        text-align: center;
        margin-top: 18px;
        border-top: 1px solid rgba(255,255,255,0.08);
        padding-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# SVG BIOLOGICAL GRAPHICS
# --------------------------------------------------------------------------------
def dna_svg(size=70):
    return f"""
    <svg class="helix-icon" width="{size}" height="{size}" viewBox="0 0 100 160" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="dnaGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#00ffc8"/>
          <stop offset="100%" stop-color="#00b4ff"/>
        </linearGradient>
      </defs>
      <path d="M20,10 C50,35 50,45 20,70 C-10,95 -10,105 20,130 C50,155 50,155 50,155"
            fill="none" stroke="url(#dnaGrad)" stroke-width="4" transform="translate(15,0)"/>
      <path d="M80,10 C50,35 50,45 80,70 C110,95 110,105 80,130 C50,155 50,155 50,155"
            fill="none" stroke="#7b61ff" stroke-width="4" transform="translate(-15,0)"/>
      <g stroke="#e3fdf9" stroke-width="2" opacity="0.8">
        <line x1="20" y1="20" x2="80" y2="20"/>
        <line x1="27" y1="40" x2="73" y2="40"/>
        <line x1="20" y1="60" x2="80" y2="60"/>
        <line x1="15" y1="80" x2="85" y2="80"/>
        <line x1="20" y1="100" x2="80" y2="100"/>
        <line x1="27" y1="120" x2="73" y2="120"/>
        <line x1="20" y1="140" x2="80" y2="140"/>
      </g>
    </svg>
    """


def rna_svg(size=60, color="#00b4ff"):
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 120 60" xmlns="http://www.w3.org/2000/svg">
      <path d="M5,30 Q20,5 35,30 T65,30 T95,30 T125,30" fill="none" stroke="{color}" stroke-width="4"/>
      <circle cx="20" cy="18" r="3.5" fill="#00ffc8"/>
      <circle cx="35" cy="30" r="3.5" fill="#ff6b6b"/>
      <circle cx="50" cy="42" r="3.5" fill="#00ffc8"/>
      <circle cx="65" cy="30" r="3.5" fill="#7b61ff"/>
      <circle cx="80" cy="18" r="3.5" fill="#ff6b6b"/>
      <circle cx="95" cy="30" r="3.5" fill="#00ffc8"/>
    </svg>
    """


def flask_svg(size=55, fill="#00ffc8"):
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
      <path d="M25 6 h14 v14 l14 30 a4 4 0 0 1 -4 6 H15 a4 4 0 0 1 -4 -6 l14 -30 Z"
            fill="none" stroke="#9fd8d0" stroke-width="2.5"/>
      <path d="M16 40 h32 l6 12 a4 4 0 0 1 -4 6 H14 a4 4 0 0 1 -4 -6 Z" fill="{fill}" opacity="0.55"/>
      <circle cx="26" cy="46" r="1.6" fill="#012"/>
      <circle cx="34" cy="50" r="1.3" fill="#012"/>
      <circle cx="30" cy="54" r="1.1" fill="#012"/>
      <line x1="22" y1="4" x2="42" y2="4" stroke="#9fd8d0" stroke-width="3" stroke-linecap="round"/>
    </svg>
    """


def heartbeat_svg(size_w=140, size_h=50, color="#ff6b6b"):
    return f"""
    <svg width="{size_w}" height="{size_h}" viewBox="0 0 200 60" xmlns="http://www.w3.org/2000/svg">
      <polyline points="0,30 30,30 40,10 50,50 60,5 70,45 80,30 200,30"
                fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
    </svg>
    """


def gauge_svg(pct, size=190, color="#ff6b6b", track="rgba(255,255,255,0.08)"):
    pct = max(0.0, min(1.0, pct))
    r = 70
    circumference = 2 * np.pi * r
    dash = circumference * pct
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 180 180" xmlns="http://www.w3.org/2000/svg">
      <circle cx="90" cy="90" r="{r}" fill="none" stroke="{track}" stroke-width="14"/>
      <circle cx="90" cy="90" r="{r}" fill="none" stroke="{color}" stroke-width="14"
              stroke-linecap="round"
              stroke-dasharray="{dash:.1f} {circumference:.1f}"
              transform="rotate(-90 90 90)"/>
      <text x="90" y="82" text-anchor="middle" font-size="30" font-weight="800" fill="{color}">{pct*100:.0f}%</text>
      <text x="90" y="106" text-anchor="middle" font-size="11" fill="#9fd8d0">Risk Score</text>
    </svg>
    """


# --------------------------------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------------------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"❌ Could not find `model.pkl` at:\n\n`{path}`\n\nMake sure it's committed to the repo.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_model()
FEATURES = list(getattr(model, "feature_names_in_", [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]))

# --------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="lab-header">
        {dna_svg(78)}
        <div>
            <div class="main-title">🧬 Diabetes Risk Lab</div>
            <div class="sub-title">AI-assisted screening powered by a K-Nearest Neighbors model</div>
        </div>
        {dna_svg(78)}
    </div>
    <div class="icon-strip">🧫 🩸 💉 🧪 🔬 🫀 📊</div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# SIDEBAR — PATIENT INPUTS
# --------------------------------------------------------------------------------
st.sidebar.markdown(f"<div style='text-align:center'>{flask_svg(60)}</div>", unsafe_allow_html=True)
st.sidebar.header("🧾 Patient Vitals")
st.sidebar.caption("Enter lab values and clinical history")

pregnancies = st.sidebar.slider("🤰 Pregnancies", 0, 17, 1, 1)
glucose = st.sidebar.slider("🩸 Glucose (mg/dL)", 40, 220, 110, 1, help="Normal fasting: <100 · Prediabetic: 100–125 · Diabetic: ≥126")
blood_pressure = st.sidebar.slider("💓 Diastolic BP (mm Hg)", 20, 130, 72, 1, help="Normal: <80 · Elevated: 80–89 · High: ≥90")
skin_thickness = st.sidebar.slider("📏 Skin Thickness (mm)", 0, 100, 23, 1, help="Triceps skinfold thickness")
insulin = st.sidebar.slider("💉 Serum Insulin (mu U/mL)", 0, 300, 80, 1, help="2-hour serum insulin; normal roughly 16–166")
bmi = st.sidebar.slider("⚖️ BMI", 10.0, 70.0, 28.0, 0.1, help="Normal: 18.5–24.9 · Overweight: 25–29.9 · Obese: ≥30")
dpf = st.sidebar.slider("🧬 Diabetes Pedigree Function", 0.05, 2.5, 0.4, 0.01, help="Estimates hereditary/genetic diabetes risk")
age = st.sidebar.slider("🎂 Age", 15, 90, 33, 1)

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button("🔬 Run Diagnostic")
st.sidebar.markdown("---")
st.sidebar.info(
    f"**Model:** KNN Classifier (k={getattr(model, 'n_neighbors', 3)})\n\n"
    f"**Training samples:** {getattr(model, 'n_samples_fit_', 'N/A')}\n\n"
    "**Dataset:** Pima Indians Diabetes"
)

raw = {
    "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
    "DiabetesPedigreeFunction": dpf, "Age": age,
}
input_df = pd.DataFrame([[raw[f] for f in FEATURES]], columns=FEATURES)

# --------------------------------------------------------------------------------
# CLINICAL FLAGGING (for the normalized bars + care plan)
# --------------------------------------------------------------------------------
def flag(value, low, high):
    """Return 'low' | 'normal' | 'high' relative to a reference band."""
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "normal"

REF = {
    "Glucose": (70, 125),
    "BloodPressure": (60, 89),
    "BMI": (18.5, 29.9),
    "Insulin": (16, 166),
    "SkinThickness": (10, 40),
    "DiabetesPedigreeFunction": (0.0, 0.6),
    "Age": (0, 45),
    "Pregnancies": (0, 6),
}
STATUS_COLOR = {"low": "#ffb347", "normal": "#00ffc8", "high": "#ff6b6b"}

# --------------------------------------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------------------------------------
col_left, col_right = st.columns([1.05, 1])

with col_left:
    st.markdown(f"<div style='text-align:center'>{rna_svg(90)}</div>", unsafe_allow_html=True)
    st.markdown("#### 🧪 Patient Panel")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    display_names = {
        "Pregnancies": "Pregnancies", "Glucose": "Glucose (mg/dL)",
        "BloodPressure": "Diastolic BP (mmHg)", "SkinThickness": "Skin Thickness (mm)",
        "Insulin": "Serum Insulin (mu U/mL)", "BMI": "BMI",
        "DiabetesPedigreeFunction": "Diabetes Pedigree Fn", "Age": "Age (yrs)",
    }
    for feat in FEATURES:
        low, high = REF[feat]
        rng = max(high - low, 1e-6)
        pct = min(max((raw[feat] - low) / rng, 0), 1) if feat not in ("Age", "Pregnancies") else min(raw[feat] / (high * 1.6 or 1), 1)
        status = flag(raw[feat], low, high)
        color = STATUS_COLOR[status]
        st.markdown(
            f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#d7f7f0;">
                <span>{display_names[feat]}</span><span style="color:{color}; font-weight:700;">{raw[feat]} ({status})</span>
              </div>
              <div style="background:rgba(255,255,255,0.08); border-radius:6px; height:8px; overflow:hidden;">
                <div style="width:{pct*100:.0f}%; background:{color}; height:8px;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center; margin-top:10px;'>{heartbeat_svg(220, 55)}</div>", unsafe_allow_html=True)

with col_right:
    st.markdown(f"<div style='text-align:center'>{flask_svg(70)}</div>", unsafe_allow_html=True)
    st.markdown("#### 🔮 Diagnostic Result")

    if predict_clicked or "last_pred" not in st.session_state:
        pred_class = int(model.predict(input_df.values)[0])
        try:
            proba = model.predict_proba(input_df.values)[0]
            risk_prob = float(proba[list(model.classes_).index(1)])
        except Exception:
            risk_prob = float(pred_class)
        st.session_state["last_pred"] = pred_class
        st.session_state["last_prob"] = risk_prob
    else:
        pred_class = st.session_state["last_pred"]
        risk_prob = st.session_state["last_prob"]

    gauge_color = "#ff6b6b" if risk_prob >= 0.5 else "#00ffc8"
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    if pred_class == 1:
        st.markdown('<span class="badge-risk">⚠️ Elevated Diabetes Risk</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-safe">✅ Low Diabetes Risk</span>', unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top:14px;'>{gauge_svg(risk_prob, color=gauge_color)}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    if risk_prob >= 0.6:
        st.error("🔴 High probability of diabetes indicators — clinical follow-up strongly advised.")
    elif risk_prob >= 0.35:
        st.warning("🟡 Borderline indicators present — monitoring recommended.")
    else:
        st.success("🟢 Indicators mostly within a healthy range.")

# --------------------------------------------------------------------------------
# PERSONALIZED CARE & PRECAUTION PLAN (rule-based, informational only)
# --------------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"<div style='text-align:center'>{rna_svg(80, '#7b61ff')}</div>", unsafe_allow_html=True)
st.markdown("### 💊 Personalized Care & Precaution Guide")
st.caption("Generated from the entered values — general lifestyle guidance only, not a medical diagnosis.")

recommendations = []

# Glucose
g_status = flag(glucose, *REF["Glucose"])
if g_status == "high":
    recommendations.append(("warn", "🩸 **Elevated glucose** — reduce refined sugars and processed carbs; favor high-fiber, low-glycemic-index foods; recheck fasting glucose and consider an HbA1c test."))
elif g_status == "low":
    recommendations.append(("warn", "🩸 **Low glucose reading** — ensure regular meals; monitor for symptoms of hypoglycemia (shakiness, sweating, dizziness)."))
else:
    recommendations.append(("good", "🩸 Glucose is within a normal range — maintain balanced meals and regular meal timing."))

# BMI
b_status = flag(bmi, *REF["BMI"])
if b_status == "high":
    recommendations.append(("warn", "⚖️ **BMI indicates overweight/obesity** — gradual weight reduction (5–7% of body weight) through calorie-aware eating and 150+ min/week of moderate exercise significantly lowers diabetes risk."))
elif b_status == "low":
    recommendations.append(("warn", "⚖️ **BMI is on the low side** — ensure adequate, balanced nutrition; consult a dietitian if unintentional."))
else:
    recommendations.append(("good", "⚖️ BMI is in a healthy range — keep up regular physical activity to maintain it."))

# Blood Pressure
bp_status = flag(blood_pressure, *REF["BloodPressure"])
if bp_status == "high":
    recommendations.append(("warn", "💓 **Elevated blood pressure** — reduce sodium intake, limit alcohol, manage stress, and monitor BP regularly; comorbid hypertension raises cardiovascular risk alongside diabetes."))
elif bp_status == "low":
    recommendations.append(("warn", "💓 Blood pressure reading is low — stay hydrated and consult a doctor if dizziness or fatigue occurs."))
else:
    recommendations.append(("good", "💓 Blood pressure is within a healthy range."))

# Insulin
ins_status = flag(insulin, *REF["Insulin"])
if ins_status == "high":
    recommendations.append(("warn", "💉 **High serum insulin** — may indicate insulin resistance; discuss with a physician, consider reducing simple-carbohydrate intake and increasing physical activity."))
elif ins_status == "low":
    recommendations.append(("warn", "💉 Low insulin reading — worth discussing with a healthcare provider, especially alongside glucose levels."))

# Age
if age >= 45:
    recommendations.append(("warn", "🎂 **Age ≥45** is itself a diabetes risk factor per ADA guidelines — annual screening (fasting glucose or HbA1c) is recommended."))
else:
    recommendations.append(("good", "🎂 Age-related risk is currently low, but healthy habits now reduce long-term risk."))

# Diabetes Pedigree Function
if dpf >= 0.6:
    recommendations.append(("warn", "🧬 **Elevated hereditary risk score** — family history raises risk regardless of current readings; earlier and more frequent screening is advisable."))

# Pregnancies
if pregnancies >= 4:
    recommendations.append(("warn", "🤰 **History of multiple pregnancies** — history of gestational diabetes (if any) warrants regular postpartum glucose screening."))

# Overall model-driven guidance
if pred_class == 1:
    recommendations.append(("warn", "🏥 **Overall model output suggests elevated risk** — please consult an endocrinologist or physician for confirmatory testing (fasting glucose, OGTT, HbA1c) before making any treatment decisions."))
else:
    recommendations.append(("good", "🏥 **Overall model output suggests low risk** — continue routine annual checkups and a balanced lifestyle."))

for kind, text in recommendations:
    st.markdown(f'<div class="care-item {kind}">{text}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="disclaimer">
    ⚕️ This tool provides statistical, informational estimates only and is <b>not a medical diagnosis</b>.
    Diabetes has no "cure" — only long-term management. Always consult a licensed healthcare professional
    for interpretation of these results and before making any changes to diet, exercise, or medication.
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="text-align:center; margin-top:20px;">
        {dna_svg(50)}
    </div>
    <p style='text-align:center; color:#5f7d78; font-size:0.8rem;'>
    Built with Streamlit &nbsp;|&nbsp; Model: KNeighborsClassifier (scikit-learn) &nbsp;|&nbsp; Diabetes Risk Lab 🧬
    </p>
    """,
    unsafe_allow_html=True,
)
