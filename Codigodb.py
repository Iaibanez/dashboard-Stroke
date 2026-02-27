import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from scipy.stats import chi2_contingency

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="Stroke Risk Dashboard",
    page_icon="🧠",
    layout="wide"
)

PRIMARY = "#1F3C88"
SECONDARY = "#2E8BC0"
ACCENT = "#F18F01"
BG = "#F4F6F9"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG};
    }}
    .card {{
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/stroke.csv")
    return df

df = load_data()

# ----------------------------
# TABS
# ----------------------------
tabs = st.tabs(["Preview", "Explore", "Study", "Tell"])

# =========================================================
# 1️⃣ PREVIEW TAB
# =========================================================
with tabs[0]:
    st.title("🧠 Stroke Risk Dataset Overview")

    st.markdown("""
    According to the World Health Organization (WHO), stroke is the 2nd leading cause 
    of death globally, responsible for approximately 11% of total deaths.

    This dataset is used to predict whether a patient is likely to get stroke 
    based on input parameters such as gender, age, diseases, and smoking status.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"<div class='card'><h3>Total Records</h3><h2>{df.shape[0]}</h2></div>",
            unsafe_allow_html=True
        )

    with col2:
        stroke_rate = df["stroke"].mean()
        st.markdown(
            f"<div class='card'><h3>Stroke Rate</h3><h2>{stroke_rate:.2%}</h2></div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"<div class='card'><h3>Total Features</h3><h2>{df.shape[1]}</h2></div>",
            unsafe_allow_html=True
        )

    st.dataframe(df.head())

# =========================================================
# 2️⃣ EXPLORE TAB
# =========================================================
with tabs[1]:
    st.title("🔍 Explore — Data Profiling")

    # Missing values
    missing_count = df.isna().sum().sum()
    missing_pct = (missing_count / (df.shape[0] * df.shape[1])) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"<div class='card'><h3>Missing Count</h3><h2>{missing_count}</h2></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<div class='card'><h3>Missing %</h3><h2>{missing_pct:.2f}%</h2></div>",
            unsafe_allow_html=True
        )

    # Age distribution
    fig_age = px.histogram(
        df,
        x="age",
        color="stroke",
        nbins=40,
        color_discrete_sequence=[SECONDARY, ACCENT],
        title="Age Distribution by Stroke"
    )
    st.plotly_chart(fig_age, use_container_width=True)

# =========================================================
# 3️⃣ STUDY TAB
# =========================================================
with tabs[2]:
    st.title("📊 Study — Relationships")

    numerical_cols = ["age", "avg_glucose_level", "bmi"]

    corr = df[numerical_cols + ["stroke"]].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # -------------------------
    # Age Slider Filter
    # -------------------------
    st.subheader("📈 Stroke Probability by Age")

    age_range = st.slider(
        "Select Age Range",
        int(df.age.min()),
        int(df.age.max()),
        (30, 80)
    )

    filtered_df = df[(df.age >= age_range[0]) & (df.age <= age_range[1])]

    age_prob = (
        filtered_df.groupby("age")["stroke"]
        .mean()
        .reset_index()
    )

    fig_prob = px.line(
        age_prob,
        x="age",
        y="stroke",
        markers=True,
        color_discrete_sequence=[PRIMARY]
    )

    fig_prob.update_layout(
        yaxis_title="Probability of Stroke",
        xaxis_title="Age"
    )

    st.plotly_chart(fig_prob, use_container_width=True)

# =========================================================
# 4️⃣ TELL TAB
# =========================================================
with tabs[3]:
    st.title("📌 Tell — Insights")

    st.markdown("""
    ### Key Insights

    - Stroke probability increases significantly with age.
    - Hypertension and heart disease show positive correlation with stroke.
    - Glucose levels tend to be higher in stroke patients.
    - Dataset is imbalanced and may require resampling for modeling.
    """)

    st.info("Next step: Modeling and predictive analytics.")
